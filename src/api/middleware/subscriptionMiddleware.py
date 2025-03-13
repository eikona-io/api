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
        self.local_cache = {}  # Local cache dictionary for stale-while-revalidate
        self.cache_ttl = 5  # Cache TTL in seconds

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

        current_time = time.time()
        # Get subscription tier from Redis with stale-while-revalidate caching
        entity_id = org_id if org_id else user_id
        redis_key = f"plan:{entity_id}"

        if redis_key in self.local_cache:
            cached_data, timestamp = self.local_cache[redis_key]
            if current_time - timestamp < self.cache_ttl:
                # print(f"[Local Cache] Using fresh cached plan_data for key: {redis_key}")
                plan_data = cached_data
            else:
                # print(f"[Local Cache] Using stale cached plan_data for key: {redis_key} and refreshing in background")
                plan_data = cached_data
                asyncio.create_task(self.refresh_cache(redis_key))
        else:
            # print(f"[Redis Fetch] Fetching plan_data from Redis for key: {redis_key}")
            plan_data = await self.redis.get(redis_key)
            if plan_data:
                self.local_cache[redis_key] = (plan_data, current_time)
                # print(f"[Local Cache] Caching plan_data for key: {redis_key}")

        if plan_data is None:
            tier = "free"
        else:
            plan_info = json.loads(plan_data)
            plans = plan_info.get("plans", [])
            tier = plans[0] if plans and len(plans) > 0 else "free"

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

    async def refresh_cache(self, redis_key: str):
        import time
        # print(f"[Background Refresh] Refreshing plan_data for key: {redis_key}")
        try:
            new_data = await self.redis.get(redis_key)
            if new_data:
                self.local_cache[redis_key] = (new_data, time.time())
                # print(f"[Background Refresh] Updated local cache for key: {redis_key}")
        except Exception as e:
            logfire.error(f"Failed to refresh cache for {redis_key}: {str(e)}")
