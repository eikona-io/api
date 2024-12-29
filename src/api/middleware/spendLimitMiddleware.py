import json
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from api.routes.types import PlanInfo
from upstash_redis.asyncio import Redis
import os
from api.database import AsyncSessionLocal
from .auth import get_current_user

logger = logging.getLogger(__name__)


class SpendLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        redis_url = os.getenv("UPSTASH_REDIS_META_REST_URL")
        redis_token = os.getenv("UPSTASH_REDIS_META_REST_TOKEN")
        self.redis = Redis(url=redis_url, token=redis_token)

    async def dispatch(self, request: Request, call_next):
        try:
            # TODO: Enable this once we fixed the override
            # if self.should_check_spend_limit(request):
            #     await self.check_spend_limit(request)
            response = await call_next(request)
            return response
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )

    def should_check_spend_limit(self, request: Request) -> bool:
        return request.method == "POST" and request.url.path in [
            "/api/run",
            "/api/session",
        ]

    async def check_spend_limit(self, request: Request):
        # Check if request.state exists and has current_user attribute
        if not hasattr(request.state, 'current_user') or request.state.current_user is None:
            return

        org_id = request.state.current_user.get("org_id")
        user_id = request.state.current_user.get("user_id")

        if org_id:
            await self.check_limit_for_entity(org_id, "org")
        elif user_id:
            await self.check_limit_for_entity(user_id, "user")
        else:
            raise HTTPException(status_code=403, detail="User or organization not found.")

    async def check_limit_for_entity(self, entity_id: str, entity_type: str):
        redis_key = f"plan:{entity_id}"
        raw_value = await self.redis.get(redis_key)
        
        
        if not raw_value:
            # Create default plan data with $5 spend limit
            default_plan = PlanInfo(spend_limit=5.0)
            raw_value = json.dumps(default_plan.model_dump())
            await self.redis.set(redis_key, raw_value)
        
        plan_data = json.loads(raw_value)
        value = PlanInfo.model_validate(plan_data)

        if "spend_limit" not in plan_data:
            updated_data = value.model_dump()
            await self.redis.set(redis_key, json.dumps(updated_data))
            logger.info(f"Updated Redis with default spend_limit for {entity_type} {entity_id}")

        if value.spent is not None and value.spent > value.spend_limit:
            logger.warning(f"Spend limit exceeded for {entity_type} {entity_id}. Spent: {value.spent}, Limit: {value.spend_limit}")
            raise HTTPException(
                status_code=403,
                detail="Spend limit exceeded. Please increase your budget limit or contact support."
            )

        logger.debug(
            f"{entity_type.capitalize()} {entity_id} - Plan: {value.plan}, Status: {value.status}, "
            f"Expires: {value.expires_at}, Spent: {value.spent}, "
            f"Spend limit: {value.spend_limit}"
        )
