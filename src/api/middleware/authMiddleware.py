import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from api.database import AsyncSessionLocal
from .auth import get_current_user

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.ignored_routes = [
            "/api/gpu_event",
            "/api/machine-built",
            "/api/fal-webhook",
            "/api/models",
            "/api/clerk/webhook",
        ]
        # print("AuthMiddleware initialized")  # Test print

    async def dispatch(self, request: Request, call_next):
        # print(f"Middleware called for: {request.method} {request.url.path}")  # Test print
        logger.info(f"Received request: {request.method} {request.url.path}")

        if self.should_authenticate(request):
            # print("Authentication required")  # Test print
            try:
                await self.authenticate(request)

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
