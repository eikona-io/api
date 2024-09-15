from typing import Optional
from sqlalchemy import select

import modal
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse  # Import JSONResponse from here
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from jose import JWTError, jwt
from sqlalchemy.orm import Session  # Import Session from SQLAlchemy
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import AsyncSessionLocal, init_db, get_db, engine
from api.routes import run, hello, internal, workflow, log, workflows, machines, models
from api.models import APIKey
from dotenv import load_dotenv
import logfire
import logging
from scalar_fastapi import get_scalar_api_reference
from fastapi.openapi.utils import get_openapi
from fastapi import APIRouter

# import all you need from fastapi-pagination
# from fastapi_pagination import Page, add_pagination, paginate
from pprint import pprint

load_dotenv()
logfire.configure()
logger = logfire
logging.basicConfig(level=logging.INFO)

# logger = logging.getLogger(__name__)

# Load environment variables from .env file

# Replace the existing oauth2_scheme declaration with this:
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        # "read:runs": "Read access to runs",
        # "write:runs": "Write access to runs",
        # Add more scopes as needed
    },
)


app = FastAPI(
    servers=[
        {"url": "https://api.comfydeploy.com/api", "description": "Production server"},
        {
            "url": "https://staging.api.comfydeploy.com/api",
            "description": "Staging server",
        },
        {"url": "http://localhost:3011/api", "description": "Local development server"},
    ]
)
# add_pagination(app)
logfire.instrument_fastapi(app)
logfire.instrument_sqlalchemy(
    engine=engine.sync_engine,
)

app.openapi_schema = None  # Clear any existing schema


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="ComfyDeploy API",
        version="1.0.0",
        description="API for ComfyDeploy",
        routes=api_router.routes,
        servers=app.servers,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
        }
    }

    # Apply Bearer Auth security globally
    openapi_schema["security"] = [{"Bearer": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# logfire.install_auto_tracing()


api_router = APIRouter()  # Remove the prefix here

# Include routers
api_router.include_router(run.router)
api_router.include_router(internal.router)
api_router.include_router(workflows.router)
api_router.include_router(workflow.router)
api_router.include_router(machines.router)
api_router.include_router(log.router)
api_router.include_router(models.router)
# api_router.include_router(hello.router)

app.include_router(api_router, prefix="/api")  # Add the prefix here instead


@app.get("/", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
        scalar_proxy_url="https://proxy.scalar.com",
        servers=[
            {"url": server["url"]} for server in app.servers
        ],  # Remove "/api" here
    )


# Get JWT secret from environment variable
JWT_SECRET = os.getenv("JWT_SECRET")
CLERK_PUBLIC_JWT_KEY = os.getenv("CLERK_PUBLIC_JWT_KEY")
ALGORITHM = "HS256"


# Function to parse JWT
async def parse_jwt(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


async def parse_clerk_jwt(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, CLERK_PUBLIC_JWT_KEY, algorithms=["RS256"])
        return payload
    except JWTError:
        return None


# Function to check if key is revoked
async def is_key_revoked(key: str, db: AsyncSession) -> bool:
    query = select(APIKey).where(APIKey.key == key, APIKey.revoked == True)
    result = await db.execute(query)
    revoked_key = result.scalar_one_or_none()
    # logger.info(f"Revoked key: {revoked_key}")
    return revoked_key is not None


# Dependency to get user data from token
async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing token")

    token = auth_header.split(" ")[1]
    user_data = await parse_jwt(token)

    if not user_data:
        user_data = await parse_clerk_jwt(token)
        # backward compatibility for old clerk tokens
        if user_data is not None:
            user_data["user_id"] = user_data["sub"]

    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # If the key has no expiration, it's not a temporary key, so we check if it's revoked
    if "exp" not in user_data:
        if await is_key_revoked(token, db):
            raise HTTPException(status_code=401, detail="Revoked token")

    return user_data


# Set up logging


# # Hide sqlalchemy logs
# logging.getLogger('sqlalchemy').setLevel(logging.ERROR)


# Modified middleware for auth check with logging
@app.middleware("http")
async def check_auth(request: Request, call_next):
    with logfire.span("api_request"):
        logger.info(f"Received request: {request.method} {request.url.path}")

        # List of routes to ignore for authentication
        ignored_routes = [
            # "/api/update-run",
            # "/api/file-upload",
            "/api/gpu_event",
            "/api/machine-built",
            "/api/fal-webhook",
        ]

        if (
            request.url.path.startswith("/api")
            and request.url.path not in ignored_routes
        ):
            try:
                async with AsyncSessionLocal() as db:
                    request.state.current_user = await get_current_user(request, db)
                # logger.info("Added current_user to request state")
            except HTTPException as e:
                logger.error(f"Authentication error: {e.detail}")
                return JSONResponse(
                    status_code=e.status_code, content={"detail": e.detail}
                )
        else:
            logger.info("Skipping auth check for non-API route or ignored route")

        response = await call_next(request)
        logger.info(f"Request completed: {response.status_code}")
        return response


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

if __name__ == "__main__":
    reload = os.getenv("ENV", "production").lower() == "development"
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, workers=4, reload=reload)
