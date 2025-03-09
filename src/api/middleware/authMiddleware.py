import logging
from api.routes.platform import get_clerk_user
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from api.database import AsyncSessionLocal
from .auth import get_current_user
from cachetools import TTLCache
from typing import Dict
from fnmatch import fnmatch
import logfire
import time  # Ensure this import is present at the top

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
            "/api/share/*",
            "/api/platform/stripe/webhook",
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
        start_time = time.time()  # Start timing here

        # print(f"Middleware called for: {request.method} {request.url.path}")  # Test print
        # logger.info(f"Received request: {request.method} {request.url.path}")

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
                logger.error(f"Authentication error: {e.detail}", extra={
                    "path": request.url.path,
                    "method": request.method,
                    "user_id": getattr(request.state, 'current_user', {}).get('user_id', 'unknown'),
                    "org_id": getattr(request.state, 'current_user', {}).get('org_id', 'unknown')
                })
                return JSONResponse(
                    status_code=e.status_code, content={"detail": e.detail}
                )
        else:
            # print("Skipping authentication")  # Test print
            logger.info("Skipping auth check for non-API route or ignored route")

        user_id = getattr(request.state, 'current_user', {}).get('user_id', 'unknown')
        org_id = getattr(request.state, 'current_user', {}).get('org_id', 'unknown')
        
        try:
            if org_id != "unknown":
                logfire.info("Organization", user_id=user_id, org_id=org_id)
            elif user_id != "unknown":
                logfire.info("User", user_id=user_id)
                
            response = await call_next(request)
            
            latency_ms = (time.time() - start_time) * 1000  # Convert latency to milliseconds
            logger.info(f"{request.method} {request.url.path} {response.status_code}", extra={
                "status_code": response.status_code,
                "path": request.url.path,
                "method": request.method,
                "user_id": user_id,
                "org_id": org_id,
                "latency_ms": latency_ms  # Explicitly log latency in milliseconds
            })
            return response
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000  # Convert latency to milliseconds even on error
            logger.error(f"Request failed: {e}", exc_info=True, extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": 500,
                "user_id": user_id,
                "org_id": org_id,
                "latency_ms": latency_ms  # Explicitly log latency in milliseconds
            })
            raise e


    def should_authenticate(self, request: Request) -> bool:
        path = request.url.path

        # Check if path matches any of the ignored routes (including wildcards)
        for route in self.ignored_routes:
            if fnmatch(path, route):
                return False

        # Require authentication for all other /api routes
        return path.startswith("/api")

    async def authenticate(self, request: Request):
        async with AsyncSessionLocal() as db:
            request.state.current_user = await get_current_user(request, db)
            
            if request.state.current_user is None:
                raise HTTPException(status_code=401, detail="Unauthorized")
        # print("User authenticated and added to request state")  # Test print
        # logger.info("Added current_user to request state")
