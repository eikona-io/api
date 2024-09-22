from .types import VolFSStructure, Model
from fastapi import HTTPException, APIRouter, Request
from typing import Any, Dict, List, Tuple
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
from sqlalchemy import update
from .types import VolFolder, VolFile
from sqlalchemy.exc import MultipleResultsFound


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Volumes"])


async def get_model_volumes(request: Request, db: AsyncSession) -> List[Dict[str, str]]:
    user_volume_query = (
        select(UserVolume)
        .apply_org_check(request)
        .where(
            UserVolume.disabled == False,
        )
    )
    result = await db.execute(user_volume_query)
    volumes = result.scalars().all()
    return [volume.to_dict() for volume in volumes]


async def get_volume_list(volume_name: str) -> VolFSStructure:
    if not volume_name:
        raise ValueError("Volume name is not provided")
    endpoint = f"{os.environ.get('MODAL_VOLUME_ENDPOINT')}/ls_full?volume_name={volume_name}&create_if_missing=true"

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.get(endpoint, headers={"Cache-Control": "no-cache"})
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
    org_id = (
        request.state.current_user["org_id"]
        if "org_id" in request.state.current_user
        else None
    )

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


async def upsert_model_to_db(
    db: AsyncSession, model_data: Dict[str, Any], request: Request
) -> None:
    user_id = request.state.current_user["user_id"]
    org_id = (
        request.state.current_user["org_id"]
        if "org_id" in request.state.current_user
        else None
    )
    user_volume_id = model_data.get("user_volume_id")
    model_name = model_data.get("name")
    folder_path = model_data.get("path")
    category = model_data.get("category")

    new_model = ModelDB(
        user_id=user_id,
        org_id=org_id,
        user_volume_id=user_volume_id,
        model_name=model_name,
        folder_path=folder_path,
        is_public=True,
        status="success",
        download_progress=100,
        upload_type="other",
        model_type="custom"
        if category
        not in [
            "checkpoint",
            "lora",
            "embedding",
            "vae",
            "clip",
            "clip_vision",
            "configs",
            "controlnet",
            "upscale_models",
            "ipadapter",
            "gligen",
            "unet",
            "custom_node",
        ]
        else category,
    )
    db.add(new_model)
    await db.commit()


async def process_volume_contents(contents, db, user_volume_id, request):
    if contents is None or len(contents) == 0:
        return
    for item in contents:
        if isinstance(item, VolFolder):
            await process_volume_contents(item.contents, db, user_volume_id, request)
        elif isinstance(item, VolFile):
            await upsert_model_to_db(
                db,
                {
                    "name": item.path.split("/")[-1],
                    "path": "/".join(item.path.split("/")[:-1]),
                    "category": item.path.split("/")[0],
                    "user_volume_id": user_volume_id,
                },
                request,
            )


async def get_private_volume_list(request: Request, db: AsyncSession) -> VolFSStructure:
    private_volumes = await get_model_volumes(request, db)

    if not private_volumes:
        private_volumes = await add_model_volume(request, db)

    if private_volumes and len(private_volumes) > 0:
        volume_structure = await get_volume_list(private_volumes[0]["volume_name"])

        # Check if models already exist in the database
        existing_models = await db.execute(
            select(ModelDB).where(ModelDB.deleted == False).apply_org_check(request)
        )
        existing_models = existing_models.scalars().all()

        # Create a set of existing model names for faster lookup
        existing_model_names = {model.model_name for model in existing_models}

        # Filter out existing models from volume_structure
        def filter_existing_models(contents):
            filtered_contents = []
            for item in contents:
                if isinstance(item, VolFolder):
                    filtered_item = item.copy()
                    filtered_item.contents = filter_existing_models(item.contents)
                    if filtered_item.contents:
                        filtered_contents.append(filtered_item)
                elif isinstance(item, VolFile):
                    if item.path.split("/")[-1] not in existing_model_names:
                        filtered_contents.append(item)
            return filtered_contents

        filtered_contents = filter_existing_models(volume_structure.contents)

        # Process and upsert models to DB
        await process_volume_contents(
            filtered_contents, db, private_volumes[0]["id"], request
        )

        return volume_structure

    return VolFSStructure(contents=[])


@async_lru_cache(expire_after=timedelta(hours=1))
async def get_public_volume_list() -> VolFSStructure:
    if not os.environ.get("SHARED_MODEL_VOLUME_NAME"):
        raise ValueError(
            "public volume name env var `SHARED_MODEL_VOLUME_NAME` is not set"
        )
    return await get_volume_list(os.environ.get("SHARED_MODEL_VOLUME_NAME"))


async def get_downloading_models(request: Request, db: AsyncSession):
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
    result = await db.execute(model_query)
    volumes = result.scalars().all()
    return volumes


async def get_private_models_from_db(
    request: Request, db: AsyncSession
) -> List[ModelDB]:
    query = (
        select(ModelDB)
        .apply_org_check(request)
        .where(
            ModelDB.deleted == False,
            ModelDB.download_progress == 100,
            ModelDB.status == "success",
        )
    )

    result = await db.execute(query)
    models = result.scalars().all()

    return models


async def get_public_models_from_db(db: AsyncSession) -> List[ModelDB]:
    query = select(ModelDB).where(
        ModelDB.deleted == False,
        ModelDB.org_id
        == os.environ.get("SHARED_MODEL_VOLUME_NAME").replace("models_", ""),
        ModelDB.download_progress == 100,
        ModelDB.status == "success",
    )

    result = await db.execute(query)
    models = result.scalars().all()

    return models


def convert_to_vol_fs_structure(models: List[ModelDB]) -> VolFSStructure:
    structure = VolFSStructure(contents=[])

    if not models:
        return structure

    categorized_models: Dict[str, VolFolder] = {}

    for model in models:
        if not model.folder_path:
            continue
        path_parts = [part for part in model.folder_path.split("/") if part]
        if not path_parts:
            continue

        # Ensure there's at least one part in path_parts
        category = path_parts[0] if path_parts else "uncategorized"

        if category not in categorized_models:
            categorized_models[category] = VolFolder(
                path=category, type="folder", contents=[]
            )
            structure.contents.append(categorized_models[category])

        current_folder = categorized_models[category]
        for i, part in enumerate(path_parts):
            new_path = "/".join(path_parts[: i + 1])
            if i == len(path_parts) - 1:
                # Add the file
                current_folder.contents.append(
                    VolFile(path=f"{new_path}/{model.model_name}", type="file")
                )
            else:
                # Find or create the next folder
                next_folder = next(
                    (
                        item
                        for item in current_folder.contents
                        if isinstance(item, VolFolder) and item.path == new_path
                    ),
                    None,
                )
                if not next_folder:
                    next_folder = VolFolder(path=new_path, type="folder", contents=[])
                    current_folder.contents.append(next_folder)
                current_folder = next_folder

    return structure


async def get_public_volume_from_db(
    db: AsyncSession,
) -> Tuple[VolFSStructure, List[ModelDB]]:
    public_volumes = await get_public_models_from_db(db)
    return convert_to_vol_fs_structure(public_volumes), public_volumes


async def get_private_volume_from_db(
    request: Request, db: AsyncSession
) -> Tuple[VolFSStructure, List[ModelDB]]:
    private_volumes = await get_private_models_from_db(request, db)
    return convert_to_vol_fs_structure(private_volumes), private_volumes


@router.get("/volume/private-models", response_model=Dict[str, Any])
async def private_models(
    request: Request, disable_cache: bool = False, db: AsyncSession = Depends(get_db)
):
    try:
        if disable_cache:
            await get_private_volume_list(request, db)
            data, models = await get_private_volume_from_db(request, db)
        else:
            data, models = await get_private_volume_from_db(request, db)
        if len(data.contents) <= 0:
            return {"structure": VolFSStructure(contents=[]), "models": []}
        return {"structure": data, "models": [model.to_dict() for model in models]}
    except Exception as e:
        logger.error(f"Error fetching private models: {str(e)}")
        logger.exception(e)  # This will log the full stack trace
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volume/public-models", response_model=Dict[str, Any])
async def public_models(
    request: Request, disable_cache: bool = False, db: AsyncSession = Depends(get_db)
):
    try:
        # if disable_cache:
        #     data = await get_public_volume_list()
        # else:
        data, models = await get_public_volume_from_db(db)
        if len(data.contents) <= 0:
            return {"structure": VolFSStructure(contents=[]), "models": []}
        return {"structure": data, "models": [model.to_dict() for model in models]}
    except Exception as e:
        logger.error(f"Error fetching public models: {str(e)}")
        logger.exception(e)  # This will log the full stack trace
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volume/downloading-models", response_model=List[Model])
async def downloading_models(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        data = await get_downloading_models(request, db)
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
