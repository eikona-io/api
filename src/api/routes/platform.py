from api.database import get_db
from api.routes.utils import fetch_user_icon, get_user_settings as get_user_settings_util
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(
    tags=["Platform"],
)


@router.get("/platform/user-settings")
async def get_user_settings(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    return await get_user_settings_util(request, db)


@router.get("/user/{user_id}")
async def get_user_meta(
    user_id: str,
):
    return await fetch_user_icon(user_id)
