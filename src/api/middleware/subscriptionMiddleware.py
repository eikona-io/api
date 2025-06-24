import json
import logging
from api.routes.platform import get_customer_plan_cached
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logfire

logger = logging.getLogger(__name__)

class SubscriptionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        try:
            if request.url.path.startswith("/api"):
                # Skip logging for update-run
                if request.url.path == "/api/update-run":
                    await self.check_subscription_access(request)
                else:
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
        if (
            not hasattr(request.state, "current_user")
            or request.state.current_user is None
        ):
            return

        org_id = request.state.current_user.get("org_id")
        user_id = request.state.current_user.get("user_id")

        # Get subscription tier from Redis with stale-while-revalidate caching
        entity_id = org_id if org_id else user_id

        plan_data = await get_customer_plan_cached(entity_id)

        # Default to free tier if no plan data
        if plan_data is None:
            tier = "free"
        else:
            try:
                plan_info = plan_data  # json.loads(plan_data)
                plans = plan_info.get("plans", [])
                tier = plans[0] if plans and len(plans) > 0 else "free"
            except (json.JSONDecodeError, TypeError, IndexError) as e:
                logfire.error(f"Error parsing plan data: {str(e)}")
                tier = "free"

        if request.url.path != "/api/update-run":
            logfire.info("Plan", tier=tier)

        if request.state is not None and request.state.current_user is not None:
            request.state.current_user["plan"] = tier