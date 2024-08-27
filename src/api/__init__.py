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
from api.routes import run, hello, internal
from api.models import APIKey
from dotenv import load_dotenv
import logfire
import logging

load_dotenv()
logfire.configure()
logger = logfire
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Load environment variables from .env file

app = FastAPI()
logfire.instrument_fastapi(app)
logfire.instrument_sqlalchemy(
    engine=engine.sync_engine,
)
# logfire.install_auto_tracing()

# Include routers
app.include_router(run.router, prefix="/api")
app.include_router(internal.router, prefix="/api")
app.include_router(hello.router)

# Initialize database
init_db()

# Get JWT secret from environment variable
JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Function to parse JWT
async def parse_jwt(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
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
    logger.info(f"Received request: {request.method} {request.url.path}")

    # List of routes to ignore for authentication
    ignored_routes = ["/api/update-run", "/api/file-upload", "/api/gpu_event", "/api/machine-built", "/api/fal-webhook"]

    if request.url.path.startswith("/api") and request.url.path not in ignored_routes:
        try:
            async with AsyncSessionLocal() as db:
                request.state.current_user = await get_current_user(request, db)
            # logger.info("Added current_user to request state")
        except HTTPException as e:
            logger.error(f"Authentication error: {e.detail}")
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
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