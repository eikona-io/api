import os
from pydantic import BaseModel
from api.database import get_db
from api.models import AuthRequest
from api.routes.utils import (
    select,
)

from api.middleware.auth import (
  generate_jwt_token
)
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import and_
from sqlalchemy.ext.asyncio import AsyncSession
import aiohttp
from datetime import datetime, timedelta, timezone
from sqlalchemy.exc import IntegrityError

router = APIRouter(
    tags=["Auth Response"],
)

async def get_clerk_user(user_id: str) -> dict:
    """
    Fetch user data from Clerk's Backend API

    Args:
        user_id: The Clerk user ID

    Returns:
        dict: User data from Clerk

    Raises:
        HTTPException: If the API call fails
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('CLERK_SECRET_KEY')}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.clerk.com/v1/users/{user_id}", headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch user data from Clerk: {await response.text()}",
            )


async def get_clerk_org(org_id: str) -> dict:
    headers = {
        "Authorization": f"Bearer {os.getenv('CLERK_SECRET_KEY')}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.clerk.com/v1/organizations/{org_id}", headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch org data from Clerk: {await response.text()}",
            )

# Local ComfyUI
@router.get("/platform/comfyui/auth-response")
async def get_local_comfyui_auth_response(
    request_id: str,
    db: AsyncSession = Depends(get_db),
):
    auth_key = (
        select(AuthRequest)
        .where(
            and_(
                AuthRequest.request_id == request_id,
                AuthRequest.expired_date > datetime.now(),
            )
        )
        .limit(1)
    )
    result = await db.execute(auth_key)
    auth_request = result.scalar_one_or_none()
    
    if not auth_request:
        raise HTTPException(status_code=404, detail="Auth request not found")
    
    # Handle timezone-aware/naive datetime comparison
    current_time = datetime.now(timezone.utc)
    created_at = auth_request.created_at

    # If created_at is naive (no timezone info), assume it's UTC
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    time_since_created = current_time - created_at
    if time_since_created.total_seconds() > 30:
        raise HTTPException(status_code=410, detail="Auth request has expired. ")
    
    user_id = auth_request.user_id
    org_id = auth_request.org_id

    user_data = await get_clerk_org(org_id) if org_id else await get_clerk_user(user_id)
    
    return {
        "api_key": auth_request.api_hash,
        "name": user_data["name"] if org_id else user_data["username"]
    }

class CreateLocalComfyuiAuthRequest(BaseModel):
    request_id: str

@router.post("/platform/comfyui/auth-request")
async def create_local_comfyui_auth_request(
    request: Request,
    body: CreateLocalComfyuiAuthRequest,
    db: AsyncSession = Depends(get_db),
):
    # Get current user info from request
    user_id = request.state.current_user["user_id"]
    org_id = request.state.current_user.get("org_id")
    
    # Set expiration date to one week from now
    expired_date = datetime.now(timezone.utc) + timedelta(weeks=1)
    
    # Generate a JWT token for API access to a week
    api_hash = generate_jwt_token(user_id, org_id, expires_in=604800)
    
    # Create the auth request
    auth_request = AuthRequest(
        request_id=body.request_id,
        user_id=user_id,
        org_id=org_id,
        api_hash=api_hash,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        expired_date=expired_date
    )
    
    try:
        # Save to database
        db.add(auth_request)
        await db.commit()
        await db.refresh(auth_request)
        
        return {
            "request_id": auth_request.request_id,
            "api_hash": auth_request.api_hash,
            "expired_date": auth_request.expired_date.isoformat(),
            "status": "success"
        }
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=400, detail="Auth request with this ID already exists")
