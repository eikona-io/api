import logging
from api.routes.platform import get_clerk_user
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from api.database import AsyncSessionLocal
from .auth import get_current_user
from cachetools import TTLCache
from typing import Dict

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._banned_cache: Dict[str, bool] = TTLCache(maxsize=100, ttl=60)
        self.ignored_routes = [
            "/api/gpu_event",
            "/api/machine-built",
            "/api/fal-webhook",
            "/api/models",
            "/api/clerk/webhook",
        ]
        # print("AuthMiddleware initialized")  # Test print

    async def get_banned_status(self, user_id: str) -> bool:
        """Get banned status with caching"""
        try:
            # Check if we have any cached value (including False/None)
            if user_id in self._banned_cache:
                return self._banned_cache[user_id]

            # Cache miss - fetch from Clerk
            user_data = await get_clerk_user(user_id)
            banned = user_data.get("banned", False)
            self._banned_cache[user_id] = banned
            return banned
        except Exception as e:
            logger.error(f"Error checking banned status: {e}")
            return False

    async def dispatch(self, request: Request, call_next):
        # print(f"Middleware called for: {request.method} {request.url.path}")  # Test print
        logger.info(f"Received request: {request.method} {request.url.path}")

        if self.should_authenticate(request):
            # print("Authentication required")  # Test print
            try:
                await self.authenticate(request)

                plan = request.state.current_user.get("plan")
                if plan == "free":
                    user_id = request.state.current_user.get("user_id")
                    banned = await self.get_banned_status(user_id)
                    if banned:
                        return JSONResponse(
                            status_code=403,
                            content={
                                "detail": "Your account has been suspended. Please contact support at founders@comfydeploy.com for assistance."
                            },
                        )

            except HTTPException as e:
                # print(f"Authentication failed: {e.detail}")  # Test print
                logger.error(f"Authentication error: {e.detail}")
                return JSONResponse(
                    status_code=e.status_code, content={"detail": e.detail}
                )
        else:
            # print("Skipping authentication")  # Test print
            logger.info("Skipping auth check for non-API route or ignored route")

        response = await call_next(request)
        # print(f"Request completed: {response.status_code}")  # Test print
        logger.info(f"Request completed: {response.status_code}")
        return response

    def should_authenticate(self, request: Request) -> bool:
        should_auth = (
            request.url.path.startswith("/api")
            and request.url.path not in self.ignored_routes
        )
        # print(f"Should authenticate: {should_auth}")  # Test print
        return should_auth

    async def authenticate(self, request: Request):
        async with AsyncSessionLocal() as db:
            request.state.current_user = await get_current_user(request, db)
        # print("User authenticated and added to request state")  # Test print
        # logger.info("Added current_user to request state")
