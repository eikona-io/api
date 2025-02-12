import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from api.database import AsyncSessionLocal
from .auth import get_current_user
from fnmatch import fnmatch

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
            "/api/share/*",
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
