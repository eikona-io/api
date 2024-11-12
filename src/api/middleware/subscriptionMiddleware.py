import json
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from upstash_redis import Redis
import os
from api.database import AsyncSessionLocal
from .auth import get_current_user

logger = logging.getLogger(__name__)

# Define pricing tiers and their allowed endpoints
PRICING_TIERS = {
    "free": {
        "allowed_endpoints": [],
        "blocked_endpoints": [
            "/api/volume/add_file",
            "/api/volume/rm",
            "/api/volume/rename_file",
            "/api/volume/list-models",
            "/api/volume/get-model-info",
        ],
    },
    "pro": {
        "allowed_endpoints": [],
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
        self.redis = Redis(url=redis_url, token=redis_token)
        self.disable_stripe = os.getenv("DISABLE_STRIPE", "false").lower() == "true"

    async def dispatch(self, request: Request, call_next):
        try:
            if request.url.path.startswith("/api"):
                if not self.disable_stripe:
                    await self.check_subscription_access(request)
            response = await call_next(request)
            return response
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code, content={"detail": exc.detail}
            )

    async def check_subscription_access(self, request: Request):
        async with AsyncSessionLocal() as db:
            request.state.current_user = await get_current_user(request, db)

        if not request.state.current_user:
            return

        org_id = request.state.current_user.get("org_id")
        user_id = request.state.current_user.get("user_id")

        # Get subscription tier from Redis
        entity_id = org_id if org_id else user_id
        redis_key = f"plan:{entity_id}"
        plan_data = self.redis.get(redis_key)

        if not plan_data:
            tier = "free"  # Default to free tier
        else:
            plan_info = json.loads(plan_data)
            tier = plan_info.get("plan", "free")

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