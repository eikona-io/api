from fastapi import Request, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt
import os
from typing import Optional
from sqlalchemy import select
from api.models import APIKey
from api.database import get_db

JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
CLERK_PUBLIC_JWT_KEY = os.getenv("CLERK_PUBLIC_JWT_KEY")

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
    # Check for cd_token in query parameters
    cd_token = request.query_params.get("cd_token")

    # Check for Authorization header
    auth_header = request.headers.get("Authorization")

    if cd_token:
        token = cd_token
    elif auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    else:
        raise HTTPException(status_code=401, detail="Invalid or missing token")

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