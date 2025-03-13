import json
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from upstash_redis.asyncio import Redis
import os
import logfire
import asyncio
import time

logger = logging.getLogger(__name__)

# Define pricing tiers and their allowed endpoints
PRICING_TIERS = {
    "free": {
        "allowed_endpoints": [],
        "blocked_endpoints": [
            "/api/volume/add_file",
            # "/api/volume/rm",
            "/api/volume/rename_file",
            "/api/volume/list-models",
            "/api/volume/get-model-info",
        ],
    },
    "pro": {
        "allowed_endpoints": [],
        "blocked_endpoints": [],
    },
    "creator": {
        "allowed_endpoints": ["*"],
        "blocked_endpoints": [],
    },
    "business": {
        "allowed_endpoints": ["*"],
        "blocked_endpoints": [],
    },
    "enterprise": {
        "allowed_endpoints": [
            "*"  # Enterprise tier has access to all endpoints
        ],
        "blocked_endpoints": [],
    },
}


class SubscriptionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        redis_url = os.getenv("UPSTASH_REDIS_META_REST_URL")
        redis_token = os.getenv("UPSTASH_REDIS_META_REST_TOKEN")
        if redis_token is None:
            print("Redis token is None", redis_url)
            self.redis = Redis(url="http://localhost:8079", token="example_token")
        else:
            self.redis = Redis(url=redis_url, token=redis_token)
        self.disable_stripe = os.getenv("DISABLE_STRIPE", "false").lower() == "true"
        self.local_cache = {}
        self.cache_ttl = 5
        self.cache_lock = asyncio.Lock()
        self.max_cache_size = 10000  # Adjust based on your expected user count

    async def dispatch(self, request: Request, call_next):
        try:
            if request.url.path.startswith("/api"):
                if not self.disable_stripe:
                    with logfire.span("Check subscription access"):
                        await self.check_subscription_access(request)
            response = await call_next(request)
            return response
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code, content={"detail": exc.detail}
            )

    async def check_subscription_access(self, request: Request):
        # Check if request.state exists and has current_user attribute
        if not hasattr(request.state, 'current_user') or request.state.current_user is None:
            return

        org_id = request.state.current_user.get("org_id")
        user_id = request.state.current_user.get("user_id")

        # Get subscription tier from Redis with stale-while-revalidate caching
        entity_id = org_id if org_id else user_id
        redis_key = f"plan:{entity_id}"

        # Get plan data with proper error handling
        plan_data = await self._get_plan_data(redis_key)

        if plan_data is None:
            logfire.info("Plan data is None, defaulting to free tier", redis_key=redis_key)
            tier = "free"
        else:
            try:
                # Ensure plan_data is properly deserialized
                if isinstance(plan_data, str):
                    plan_info = json.loads(plan_data)
                else:
                    plan_info = plan_data

                plans = plan_info.get("plans", [])

                # Log the raw plan_info for debugging
                logfire.info("Plan info retrieved",
                             redis_key=redis_key,
                             has_plans=bool(plans),
                             plans_length=len(plans),
                             plans=plans)

                tier = plans[0] if plans and len(plans) > 0 else "free"
                if not plans or len(plans) == 0:
                    logfire.warning("Empty plans array in Redis data", redis_key=redis_key, plan_info=plan_info)
            except Exception as e:
                logfire.error("Error parsing plan data",
                              redis_key=redis_key,
                              error=str(e),
                              plan_data=plan_data)
                tier = "free"

        logfire.info("Plan", tier=tier)

        # Check if endpoint is blocked or not allowed for the tier
        if self.is_endpoint_blocked(
            request.url.path, tier
        ) or not self.is_endpoint_allowed(request.url.path, tier):
            logger.warning(f"Access denied for {request.url.path} in {tier} tier")
            raise HTTPException(
                status_code=403,
                detail=f"This endpoint is not available in your current {tier} tier. Please upgrade your subscription.",
            )

        if request.state is not None and request.state.current_user is not None:
            request.state.current_user["plan"] = tier

    def is_endpoint_allowed(self, path: str, tier: str) -> bool:
        tier_config = PRICING_TIERS.get(tier, PRICING_TIERS["free"])
        allowed_endpoints = tier_config["allowed_endpoints"]

        # If * is in allowed_endpoints, allow everything
        if "*" in allowed_endpoints:
            return True

        # If no specific allowed endpoints are defined, allow it
        if not allowed_endpoints:
            return True

        # Check if any allowed endpoint matches the path
        for allowed_path in allowed_endpoints:
            # Exact match
            if path == allowed_path:
                return True
            # Path starts with allowed path and next char is /
            if path.startswith(allowed_path + "/"):
                return True

        return False

    def is_endpoint_blocked(self, path: str, tier: str) -> bool:
        tier_config = PRICING_TIERS.get(tier, PRICING_TIERS["free"])
        blocked_endpoints = tier_config["blocked_endpoints"]
        allowed_endpoints = tier_config["allowed_endpoints"]

        for blocked_path in blocked_endpoints:
            # Exact match
            if path == blocked_path:
                return True

            # If blocked path is a parent path (e.g. /api/volume)
            # Block all child paths unless specifically allowed
            if path.startswith(blocked_path + "/"):
                # Check if there's a more specific allowed endpoint that overrides this block
                for allowed_path in allowed_endpoints:
                    if allowed_path.startswith(blocked_path + "/") and path.startswith(
                        allowed_path
                    ):
                        return False
                return True

        return False

    async def _get_plan_data(self, redis_key: str):
        """Get plan data with proper caching and error handling"""
        current_time = time.time()
        plan_data = None

        try:
            async with self.cache_lock:
                if redis_key in self.local_cache:
                    cached_data, timestamp = self.local_cache[redis_key]
                    if current_time - timestamp < self.cache_ttl:
                        logfire.debug("Using fresh cached plan_data", redis_key=redis_key)
                        plan_data = cached_data
                    else:
                        logfire.debug("Using stale cached plan_data and refreshing", redis_key=redis_key)
                        plan_data = cached_data
                        asyncio.create_task(self.refresh_cache(redis_key))
                else:
                    logfire.debug("Fetching plan_data from Redis", redis_key=redis_key)
                    plan_data = await self.redis.get(redis_key)

                    # Ensure plan_data is properly deserialized if it's a string
                    if plan_data and isinstance(plan_data, str):
                        try:
                            # Verify it's valid JSON before caching
                            json_data = json.loads(plan_data)
                            # Store the parsed JSON object instead of the string
                            plan_data = json_data
                        except json.JSONDecodeError as e:
                            logfire.error("Invalid JSON in Redis", redis_key=redis_key, error=str(e))

                    if plan_data:
                        # Cache management
                        if len(self.local_cache) >= self.max_cache_size:
                            oldest_keys = sorted(
                                self.local_cache.items(), key=lambda x: x[1][1]
                            )[: max(1, self.max_cache_size // 10)]

                            for old_key, _ in oldest_keys:
                                del self.local_cache[old_key]

                        # Cache the data
                        self.local_cache[redis_key] = (plan_data, current_time)
                        logfire.debug("Cached plan_data", redis_key=redis_key)
        except Exception as e:
            logfire.error("Error retrieving plan data", redis_key=redis_key, error=str(e))

        return plan_data

    async def refresh_cache(self, redis_key: str):
        try:
            new_data = await self.redis.get(redis_key)

            # Ensure new_data is properly deserialized if it's a string
            if new_data and isinstance(new_data, str):
                try:
                    # Verify it's valid JSON before caching
                    json_data = json.loads(new_data)
                    # Store the parsed JSON object instead of the string
                    new_data = json_data
                except json.JSONDecodeError as e:
                    logfire.error("Invalid JSON during cache refresh", redis_key=redis_key, error=str(e))
                    return

            if new_data:
                async with self.cache_lock:
                    self.local_cache[redis_key] = (new_data, time.time())
                    logfire.debug("Refreshed cache", redis_key=redis_key)
        except Exception as e:
            logfire.error("Failed to refresh cache", redis_key=redis_key, error=str(e))
