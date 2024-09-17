from .types import VolFSStructure, Model
from fastapi import HTTPException, APIRouter, Request
from typing import Any, Dict, List
import logging
import os
import httpx
from .utils import async_lru_cache, select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from api.database import get_db
from api.models import Model as ModelDB, UserVolume
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Models"])


async def get_model_volumes(request: Request, db: AsyncSession) -> List[Dict[str, str]]:
    model_query = (
        select(ModelDB)
        .order_by(ModelDB.model_name.desc())
        .apply_org_check(request)
        .where(
            ModelDB.deleted == False,
        )
    )
    result = await db.execute(model_query)
    volumes = result.scalars().all()
    return [
        {"volume_name": "models_" + (volume.org_id or volume.user_id)}
        for volume in volumes
    ]


async def getVolumeList(volume_name: str) -> VolFSStructure:
    if not volume_name:
        raise ValueError("Volume name is not provided")
    endpoint = f"{os.environ.get('MODAL_VOLUME_ENDPOINT')}/ls_full?volume_name={volume_name}&create_if_missing=true"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(endpoint, headers={"Cache-Control": "no-cache"})
            print("response", response)
            response.raise_for_status()
            data = response.json()
            return VolFSStructure(**data)
        except httpx.ReadTimeout:
            logger.error(f"Timeout error while fetching volume list for {volume_name}")
            raise HTTPException(status_code=504, detail="Request timed out")
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error {e.response.status_code} while fetching volume list for {volume_name}"
            )
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            logger.error(
                f"Unexpected error while fetching volume list for {volume_name}: {str(e)}"
            )
            raise HTTPException(status_code=500, detail="Internal server error")


async def add_model_volume(request: Request, db: AsyncSession) -> Dict[str, Any]:
    user_id = request.state.current_user["user_id"]
    org_id = request.state.current_user["org_id"] if "org_id" in request.state.current_user else None

    # Insert new volume
    new_volume = UserVolume(
        user_id=user_id,
        org_id=org_id,
        volume_name=f"models_{org_id if org_id else user_id}",
        disabled=False,
    )
    db.add(new_volume)
    await db.commit()
    await db.refresh(new_volume)

    return new_volume.to_dict()


async def set_initial_user_data(user_id: str):
    # Implement the logic to set initial user data
    # This is a placeholder and should be replaced with actual implementation
    pass


async def getPrivateVolumeList(request: Request, db: AsyncSession) -> VolFSStructure:
    private_volumes = await get_model_volumes(request, db)

    if not private_volumes:
        private_volumes = await add_model_volume(request, db)

    if private_volumes and len(private_volumes) > 0:
        return await getVolumeList(private_volumes[0]["volume_name"])

    return VolFSStructure(contents=[])

@async_lru_cache(expire_after=timedelta(hours=1))
async def getPublicVolumeList() -> VolFSStructure:
    if not os.environ.get("SHARED_MODEL_VOLUME_NAME"):
        raise ValueError(
            "public volume name env var `SHARED_MODEL_VOLUME_NAME` is not set"
        )
    return await getVolumeList(os.environ.get("SHARED_MODEL_VOLUME_NAME"))


async def getDownloadingModels(request: Request, db: AsyncSession):
    model_query = (
        select(ModelDB)
        .order_by(ModelDB.model_name.desc())
        .apply_org_check(request)
        .where(
            ModelDB.deleted == False,
            ModelDB.download_progress != 100,
            ModelDB.status != "failed",
            ModelDB.created_at > datetime.now() - timedelta(hours=1),
        )
    )
    print("model_query", model_query)
    result = await db.execute(model_query)
    print("result", result)
    volumes = result.scalars().all()
    return volumes


@router.get("/models/private-models", response_model=VolFSStructure)
async def private_models(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        data = await getPrivateVolumeList(request, db)
        return data
    except Exception as e:
        logger.error(f"Error fetching private models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/public-models", response_model=VolFSStructure)
async def public_models(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        data = await getPublicVolumeList()
        return data
    except Exception as e:
        logger.error(f"Error fetching public models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/downloading-models", response_model=List[Model])
async def downloading_models(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        data = await getDownloadingModels(request, db)
        model_data = []
        for model in data:
            model_dict = model.to_dict()
            logger.info(f"download_progress: {model.download_progress}")
            if model.download_progress == 100:
                model_dict["is_done"] = True
            model_data.append(model_dict)
        return JSONResponse(content=model_data)
    except Exception as e:
        logger.error(f"Error fetching downloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
