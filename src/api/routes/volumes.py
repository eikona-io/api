import time
from .types import VolFSStructure, Model
from fastapi import HTTPException, APIRouter, Request, BackgroundTasks
from typing import Any, Dict, List, Tuple, Union, Optional
import logging
import os
import httpx
from .utils import async_lru_cache, get_user_settings, select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from api.database import get_db
from api.models import Model as ModelDB, UserVolume
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from .types import VolFolder, VolFile
from sqlalchemy.exc import MultipleResultsFound
from pydantic import BaseModel
from enum import Enum
import re
import grpclib
from modal import Volume, Secret
import modal
from huggingface_hub import HfApi
import logfire

# import aiohttp
# from modal_downloader.modal_downloader import modal_download_file_task, modal_downloader_app
import json
import aiohttp
from urllib.parse import urlparse
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Volumes"])


async def retrieve_model_volumes(
    request: Request, db: AsyncSession
) -> List[Dict[str, str]]:
    volumes = await get_model_volumes(request, db)
    if len(volumes) == 0:
        volumes = [await add_model_volume(request, db)]
    return volumes


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


async def get_volume_list(request: Request, volume_name: str) -> VolFSStructure:
    if not volume_name:
        raise ValueError("Volume name is not provided")

    try:
        response = await volume_full(request, volume_name, create_if_missing=True)
        return VolFSStructure(**response)
    except HTTPException as e:
        logger.error(
            f"HTTP error {e.status_code} while fetching volume list for {volume_name}"
        )
        raise
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
    size = model_data.get("size")

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
        size=size,
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
    if model_data.get("id") is not None:
        new_model.id = model_data.get("id")

    with logfire.span("Upserting model", output=new_model):
        query = insert(ModelDB).values(new_model.to_dict())
        # update_dict = {c.name: c for c in query.excluded if not c.primary_key}
        query = query.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "model_name": new_model.model_name,
                "folder_path": new_model.folder_path,
                "download_progress": 100,
                "size": new_model.size,
                "updated_at": datetime.now(),
            },
        )
        # logger.info(f"model_name: {model_name}, size: {new_model.size}")
        await db.execute(query)
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
                    "id": item.id,
                    "name": item.path.split("/")[-1],
                    "path": "/".join(item.path.split("/")[:-1]),
                    "category": item.path.split("/")[0],
                    "user_volume_id": user_volume_id,
                    "size": item.size,
                },
                request,
            )


async def refresh_db_files_from_volume(
    request: Request, db: AsyncSession
) -> VolFSStructure:
    private_volumes = await get_model_volumes(request, db)

    if not private_volumes:
        private_volumes = await add_model_volume(request, db)

    if private_volumes and len(private_volumes) > 0:
        volume_structure = await get_volume_list(
            request, private_volumes[0]["volume_name"]
        )

        # Check if models already exist in the database
        existing_models = await db.execute(
            select(ModelDB).where(ModelDB.deleted == False).apply_org_check(request)
        )
        existing_models = existing_models.scalars().all()

        # Create a set of existing model names for faster lookup
        existing_model_names = {
            model.folder_path + "/" + model.model_name: model
            for model in existing_models
            if model.folder_path is not None and model.model_name is not None
        }

        # print("existing_model_names: ", existing_model_names)

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
                    # print(item.path)
                    # logger.info(f"existing_model_names {existing_model_names}")
                    # logger.info(
                    #     f"item.path, {item.path}, {item.path not in existing_model_names}"
                    # )
                    # If the size is None, probably is old file, we should add it to the list
                    if item.path not in existing_model_names or (
                        existing_model_names[item.path].size is None
                        and item.size is not None
                    ):
                        if item.path in existing_model_names:
                            item.id = existing_model_names[item.path].id
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
async def get_public_volume_list(request: Request) -> VolFSStructure:
    if not os.environ.get("SHARED_MODEL_VOLUME_NAME"):
        raise ValueError(
            "public volume name env var `SHARED_MODEL_VOLUME_NAME` is not set"
        )
    return await get_volume_list(request, os.environ.get("SHARED_MODEL_VOLUME_NAME"))


async def get_downloading_models(request: Request, db: AsyncSession):
    model_query = (
        select(ModelDB)
        .order_by(ModelDB.model_name.desc())
        .apply_org_check(request)
        .where(
            ModelDB.deleted == False,
            ModelDB.download_progress != 100,
            ModelDB.status != "failed",
            ModelDB.created_at > datetime.now() - timedelta(hours=24),
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
            ModelDB.model_name != None,
            ModelDB.folder_path != None,
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

    def create_or_get_folder(
        path: str, parent: Union[VolFSStructure, VolFolder]
    ) -> VolFolder:
        for item in parent.contents:
            if isinstance(item, VolFolder) and item.path == path:
                return item
        new_folder = VolFolder(path=path, type="folder", contents=[])
        parent.contents.append(new_folder)
        return new_folder

    for model in models:
        if not model.folder_path:
            continue
        path_parts = [part for part in model.folder_path.split("/") if part]
        if not path_parts:
            continue

        current_folder = structure
        for part in path_parts:
            current_folder = create_or_get_folder(part, current_folder)

        file_path = os.path.join(model.folder_path, model.model_name)
        current_folder.contents.append(
            VolFile(path=file_path, type="file", size=model.size)
        )

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
            await refresh_db_files_from_volume(request, db)
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


# Type definitions
class RenameFileBody(BaseModel):
    volume_name: str
    src_path: str
    new_filename: str
    overwrite: Optional[bool] = False


class RemoveFileInput(BaseModel):
    path: str
    volume_name: str


class AddFileInput(BaseModel):
    # volume_name: str
    download_url: str
    folder_path: str
    filename: str
    # callback_url: str
    db_model_id: str
    upload_type: str


class ModelDownloadStatus(Enum):
    PROGRESS = "progress"
    SUCCESS = "success"
    FAILED = "failed"


# Constants
FILE_TYPE = 1
DIRECTORY_TYPE = 2

# New routes and helper functions


class NewRenameFileBody(BaseModel):
    filename: str
    
class AddFileInputNew(BaseModel):
    url: str
    filename: Optional[str]
    folder_path: str


@router.post("/file")
async def add_file(
    request: Request, 
    body: AddFileInputNew,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Handle model file uploads from different sources (Civitai, HuggingFace, or generic URLs)"""
    
    if "civitai.com/models/" in body.url:
        return await handle_civitai_model(request, body, db, background_tasks)
    elif "huggingface.co/" in body.url:
        return await handle_huggingface_model(request, body, db, background_tasks)
    else:
        return await handle_generic_model(request, body, db, background_tasks)

@router.post("/file/{file_id}/rename")
async def rename_file(
    request: Request,
    body: NewRenameFileBody,
    file_id: str,
    db: AsyncSession = Depends(get_db),
):
    new_filename = body.filename
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    volume_name = f"models_{org_id}" if org_id else f"models_{user_id}"

    volume = Volume.from_name(volume_name)

    model = (
        await db.execute(
            select(ModelDB)
            .apply_org_check(request)
            .where(ModelDB.id == file_id, ~ModelDB.deleted)
        )
    ).scalar_one()

    src_path = os.path.join(model.folder_path, model.model_name)

    # check src_path is a file
    is_valid, error_message = await validate_file_path_aio(src_path, volume)
    if not is_valid:
        if "not found" in error_message:
            raise HTTPException(status_code=404, detail=error_message)
        else:
            raise HTTPException(status_code=400, detail=error_message)

    filename_valid = is_valid_filename(new_filename)
    if not filename_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename{new_filename}, only allow characters, numerics, underscores, and hyphens.",
        )

    folder_path = os.path.dirname(src_path)
    dst_path = os.path.join(folder_path, new_filename)

    # Check if the destination file exists and if we can overwrite it
    try:
        contents = await volume.listdir.aio(dst_path)
        if contents:
            # if not overwrite:
            raise HTTPException(
                status_code=400,
                detail="Destination file exists and overwrite is False.",
            )
    except Exception as _:
        pass

    print("src_path: ", src_path)
    print("dst_path: ", dst_path)

    await volume.copy_files.aio([src_path], dst_path)
    await volume.remove_file.aio(src_path)

    model.model_name = new_filename

    await db.commit()

    return model.to_dict()


@router.delete("/file/{file_id}")
async def delete_file(
    request: Request, file_id: str, db: AsyncSession = Depends(get_db)
):
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    volume_name = f"models_{org_id}" if org_id else f"models_{user_id}"

    model = (
        await db.execute(
            select(ModelDB)
            .apply_org_check(request)
            .where(ModelDB.id == file_id, ~ModelDB.deleted)
        )
    ).scalar_one()

    volume = Volume.from_name(volume_name)
    src_path = os.path.join(model.folder_path, model.model_name)

    is_valid, error_message = await validate_file_path_aio(src_path, volume)
    if not is_valid:
        if "not found" in error_message:
            raise HTTPException(status_code=404, detail=error_message)
        else:
            raise HTTPException(status_code=400, detail=error_message)

    await volume.remove_file.aio(src_path)

    model.deleted = True

    await db.commit()

    return model.to_dict()


# Used by v1 dashboard
@router.post("/volume/rename_file", deprecated=True, include_in_schema=False)
async def rename_file_old(request: Request, body: RenameFileBody):
    src_path = body.src_path
    new_filename = body.new_filename
    overwrite = body.overwrite
    volume_name = body.volume_name

    print("rename_file", body)

    try:
        volume = lookup_volume(volume_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # check src_path is a file
    is_valid, error_message = validate_file_path(src_path, volume)
    if not is_valid:
        if "not found" in error_message:
            raise HTTPException(status_code=404, detail=error_message)
        else:
            raise HTTPException(status_code=400, detail=error_message)

    filename_valid = is_valid_filename(new_filename)
    if not filename_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename{new_filename}, only allow characters, numerics, underscores, and hyphens.",
        )

    folder_path = os.path.dirname(src_path)
    dst_path = os.path.join(folder_path, new_filename)

    # Check if the destination file exists and if we can overwrite it
    try:
        contents = volume.listdir(dst_path)
        if contents:
            if not overwrite:
                raise HTTPException(
                    status_code=400,
                    detail="Destination file exists and overwrite is False.",
                )
    except Exception as _:
        pass

    volume.copy_files([src_path], dst_path)
    volume.remove_file(src_path)

    return {
        "old_path": src_path,
        "new_path": dst_path,
    }


@router.post("/volume/rm", deprecated=True, include_in_schema=False)
async def remove_file_old(request: Request, body: RemoveFileInput):
    try:
        volume = lookup_volume(body.volume_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    is_valid, error_message = validate_file_path(body.path, volume)
    if not is_valid:
        if "not found" in error_message:
            raise HTTPException(status_code=404, detail=error_message)
        else:
            raise HTTPException(status_code=400, detail=error_message)

    volume.remove_file(body.path)
    return {"deleted_path": body.path}


async def handle_file_download(
    request: Request,
    db: AsyncSession,
    download_url: str,
    folder_path: str,
    filename: str,
    upload_type: str,
    db_model_id: str,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """Helper function to handle file downloads with progress tracking"""
    try:
        current_user = request.state.current_user
        user_id = current_user["user_id"]
        org_id = current_user["org_id"] if "org_id" in current_user else None

        volumes = await retrieve_model_volumes(request, db)
        volume_name = volumes[0]["volume_name"]

        user_settings = await get_user_settings(request, db)
        hugging_face_token = os.environ.get("HUGGINGFACE_TOKEN")
        if user_settings is not None and user_settings.hugging_face_token:
            hugging_face_token = (
                user_settings.hugging_face_token.strip() or hugging_face_token
            )

        volume = lookup_volume(volume_name, create_if_missing=True)
        full_path = os.path.join(folder_path, filename)

        try:
            file_exists = does_file_exist(full_path, volume)
            if file_exists:
                raise HTTPException(status_code=400, detail="File already exists.")
        except grpclib.exceptions.GRPCError as e:
            print("e: ", str(e))
            raise HTTPException(status_code=400, detail="Error: " + str(e))

        modal_download_file_task = modal.Function.lookup(
            "volume-operations", "modal_download_file_task"
        )

        async def event_generator():
            try:
                async for event in modal_download_file_task.remote_gen.aio(
                    download_url,
                    folder_path,
                    filename,
                    db_model_id,
                    full_path,
                    volume_name,
                    upload_type,
                    hugging_face_token,
                ):
                    # Update database with the event status
                    if event.get("status") == "progress":
                        model_status_query = (
                            update(ModelDB)
                            .where(ModelDB.id == db_model_id)
                            .values(
                                updated_at=datetime.now(),
                                download_progress=event.get("download_progress", 0),
                            )
                        )
                    elif event.get("status") == "success":
                        model_status_query = (
                            update(ModelDB)
                            .where(ModelDB.id == db_model_id)
                            .values(
                                status="success",
                                updated_at=datetime.now(),
                                download_progress=100,
                            )
                        )
                    elif event.get("status") == "failed":
                        model_status_query = (
                            update(ModelDB)
                            .where(ModelDB.id == db_model_id)
                            .values(
                                status="failed",
                                error_log=event.get("error_log"),
                                updated_at=datetime.now(),
                            )
                        )

                    await db.execute(model_status_query)
                    await db.commit()

                    # yield json.dumps(event) + "\n"
            except Exception as e:
                error_event = {
                    "status": "failed",
                    "error_log": str(e),
                    "model_id": db_model_id,
                    "download_progress": 0,
                }
                # yield json.dumps(error_event) + "\n"

                model_status_query = (
                    update(ModelDB)
                    .where(ModelDB.id == db_model_id)
                    .values(
                        status="failed",
                        error_log=str(e),
                        updated_at=datetime.now(),
                    )
                )
                await db.execute(model_status_query)
                await db.commit()
                raise e

        background_tasks.add_task(event_generator)

        return JSONResponse(content={"message": "success"})

    except Exception as e:
        print(f"Error in file download: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/volume/add_file", deprecated=True, include_in_schema=False)
async def add_file_old(
    request: Request,
    body: AddFileInput,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    return await handle_file_download(
        request=request,
        db=db,
        download_url=body.download_url,
        folder_path=body.folder_path,
        filename=body.filename,
        upload_type=body.upload_type,
        db_model_id=body.db_model_id,
        background_tasks=background_tasks,
    )

async def handle_generic_model(
    request: Request,
    data: AddFileInputNew, 
    db: AsyncSession,
    background_tasks: BackgroundTasks
):
    filename = data.filename or await get_filename_from_url(data.url)
    if not filename:
        model = await create_model_error_record(
            request=request,
            db=db,
            url=data.url,
            error_message="filename not found",
            upload_type="download-url",
            folder_path=data.folder_path
        )
        raise HTTPException(status_code=400, detail="No filename found")

    # Check if file exists
    await check_file_existence(filename, data.folder_path, data.url, "download-url", request, db)

    volumes = await retrieve_model_volumes(request, db)
    
    model = await add_model_download_url(
        request=request,
        db=db,
        upload_type="download-url",
        model_name=filename,
        url=data.url,
        volume_id=volumes[0]["id"],
        custom_path=data.folder_path
    )

    await handle_file_download(
        request=request,
        db=db,
        download_url=data.url,
        folder_path=data.folder_path,
        filename=filename,
        upload_type="download-url", 
        db_model_id=str(model.id),
        background_tasks=background_tasks
    )

    return {"message": "Generic model download started"}

# Helper functions

async def add_model_download_url(
    request: Request,
    db: AsyncSession,
    upload_type: str,
    model_name: str,
    url: str,
    volume_id: str,
    custom_path: str
) -> ModelDB:
    """Create a new model record in the database"""
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    new_model = ModelDB(
        user_id=user_id,
        org_id=org_id,
        user_volume_id=volume_id,
        upload_type=upload_type,
        model_name=model_name,
        user_url=url,
        model_type="custom",
        folder_path=custom_path,
        status="started",
        download_progress=0
    )

    db.add(new_model)
    await db.commit()
    await db.refresh(new_model)
    
    return new_model

async def create_model_error_record(
    request: Request,
    db: AsyncSession,
    url: str,
    error_message: str,
    upload_type: str,
    folder_path: str,
    model_name: Optional[str] = None
) -> ModelDB:
    """Create an error record in the model table"""
    volumes = await retrieve_model_volumes(request, db)
    current_user = request.state.current_user
    
    new_model = ModelDB(
        user_id=current_user["user_id"],
        org_id=current_user.get("org_id"),
        user_volume_id=volumes[0]["id"],
        upload_type=upload_type,
        model_type="custom",
        user_url=url if upload_type == "download-url" else None,
        civitai_url=url if upload_type == "civitai" else None,
        error_log=error_message,
        folder_path=folder_path,
        model_name=model_name,
        status="failed"
    )

    db.add(new_model)
    await db.commit()
    await db.refresh(new_model)
    
    return new_model

@router.post("/volume/file/{file_id}/retry", include_in_schema=False)
@router.post("/file/{file_id}/retry")
async def retry_download(
    request: Request,
    file_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    # Query to get the existing model record
    query = (
        select(ModelDB)
        .where(
            ModelDB.id == file_id,
            ModelDB.deleted == False,
            ModelDB.status == "failed",  # Only allow retrying failed downloads
        )
        .apply_org_check(request)
    )

    result = await db.execute(query)
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=404, detail="Model not found or not eligible for retry"
        )

    # Use the helper function for the download
    res = await handle_file_download(
        request=request,
        db=db,
        download_url=model.user_url,
        folder_path=model.folder_path,
        filename=model.model_name,
        upload_type=model.upload_type,
        db_model_id=str(model.id),
        background_tasks=background_tasks,
    )

    # Reset the model status for retry
    model_status_query = (
        update(ModelDB)
        .where(ModelDB.id == file_id)
        .values(
            status="started",
            error_log=None,
            updated_at=datetime.now(),
            download_progress=0,
        )
    )
    await db.execute(model_status_query)
    await db.commit()

    return res


# Helper functions
def does_file_exist(path: str, volume: Volume) -> bool:
    try:
        contents = volume.listdir(path)
        if not contents:
            return False
        if len(contents) == 1 and contents[0].type == FILE_TYPE:
            return True
        return False
    except grpclib.exceptions.GRPCError as e:
        if e.status == grpclib.Status.NOT_FOUND:
            return False
        else:
            raise e


async def validate_file_path_aio(path: str, volume: Volume):
    try:
        contents = await volume.listdir.aio(path)
        if not contents:
            return False, "No file found or the first item is not a file."
        if len(contents) > 1:
            return False, "directory supplied"
        if contents[0].type == DIRECTORY_TYPE:
            return False, "directory supplied"
        if contents[0].type != FILE_TYPE:
            return False, "not a file"
        return True, None
    except grpclib.exceptions.GRPCError as e:
        if e.status == grpclib.Status.NOT_FOUND:
            return False, f"path: {path} not found."
        else:
            return False, str(e)


def validate_file_path(path: str, volume: Volume):
    try:
        contents = volume.listdir(path)
        if not contents:
            return False, "No file found or the first item is not a file."
        if len(contents) > 1:
            return False, "directory supplied"
        if contents[0].type == DIRECTORY_TYPE:
            return False, "directory supplied"
        if contents[0].type != FILE_TYPE:
            return False, "not a file"
        return True, None
    except grpclib.exceptions.GRPCError as e:
        if e.status == grpclib.Status.NOT_FOUND:
            return False, f"path: {path} not found."
        else:
            return False, str(e)


def is_valid_filename(filename):
    pattern = r"^[\w\-\.]+$"
    if re.match(pattern, filename):
        return True
    else:
        return False


def lookup_volume(volume_name: str, create_if_missing: bool = False):
    try:
        return Volume.lookup(volume_name, create_if_missing=create_if_missing)
    except Exception as e:
        raise Exception(f"Can't find Volume: {e}")


@router.get("/volume/ls_full", deprecated=True, include_in_schema=False)
async def volume_full(
    request: Request, volume_name: str, create_if_missing: bool = False
):
    try:
        volume = lookup_volume(volume_name, create_if_missing=create_if_missing)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    contents = volume.listdir("/", recursive=True)
    try:
        transformed_contents = []
        for content in contents:
            entry_data = {
                "path": content.path,
                "type": "folder" if content.type == DIRECTORY_TYPE else "file",
                # "mtime": content.mtime,
                "size": content.size if content.type == FILE_TYPE else None,
                "contents": [] if content.type == DIRECTORY_TYPE else None,
            }

            # print("entry_data: ", entry_data)

            # Simulate the nested structure that recursive_listdir would create
            path_parts = content.path.split("/")
            current_level = transformed_contents
            for part in path_parts[:-1]:
                found = False
                for item in current_level:
                    if item["path"] == part and item["type"] == "folder":
                        current_level = item["contents"]
                        found = True
                        break
                if not found:
                    new_folder = {"path": part, "type": "folder", "contents": []}
                    current_level.append(new_folder)
                    current_level = new_folder["contents"]

            current_level.append(entry_data)

        return {"contents": transformed_contents}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while listing directory contents: {str(e)}",
        )


@router.get("/volume/ls", include_in_schema=False)
async def list_contents(request: Request, volume_name: str, path: str = "/"):
    try:
        volume = lookup_volume(volume_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        contents = volume.listdir(path)
        mapped_contents = [
            {
                "path": entry.path,
                "type": entry.type,
                "mtime": entry.mtime,
                "size": entry.size,
            }
            for entry in contents
        ]
        contents = mapped_contents
    except grpclib.exceptions.GRPCError as e:
        if e.status == grpclib.Status.NOT_FOUND:
            raise HTTPException(status_code=404, detail="Path not found.")
        else:
            raise HTTPException(status_code=500, detail="Internal server error.")
    return {"contents": contents}


class StatusEnum(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PROGRESS = "progress"


class RequestModel(BaseModel):
    model_id: str
    status: StatusEnum
    error_log: Optional[str] = None
    filehash_sha256: Optional[str] = None
    download_progress: Optional[float] = None


@router.post("/volume/volume-upload", include_in_schema=False)
async def update_status(
    request: Request, body: RequestModel, db: AsyncSession = Depends(get_db)
):
    import requests

    try:
        # Convert the Pydantic model to a dictionary
        body_dict = body.dict()

        # Convert Enum to string
        body_dict["status"] = body_dict["status"].value

        response = requests.post(
            "http://127.0.0.1:3010/api/volume-upload", json=body_dict
        )
        response.raise_for_status()  # Raise an exception for HTTP errors

        return {"message": "Status updated successfully"}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error updating status: {str(e)}")
    # try:
    #     # Assuming you have a database connection and model table defined
    #     # You'll need to replace these with your actual database operations
    #     if status == StatusEnum.SUCCESS:
    #         # Update model status to success
    #         model_status_query = (
    #             update(ModelDB)
    #             .where(ModelDB.id == model_id)
    #             .values(
    #                 status="success",
    #                 updated_at=datetime.now(),
    #                 download_progress=100,
    #                 filehash_sha256=filehash_sha256,
    #             )
    #         )
    #         result = await db.execute(model_status_query)
    #         pass
    #     elif status == StatusEnum.PROGRESS:
    #         # Update model download progress
    #         model_status_query = (
    #             update(ModelDB)
    #             .where(ModelDB.id == model_id)
    #             .values(
    #                 updated_at=datetime.now(),
    #                 download_progress=download_progress,
    #             )
    #         )
    #         result = await db.execute(model_status_query)
    #         print("download_progress: ", download_progress)
    #         pass
    #     elif status == StatusEnum.FAILED:
    #         # Update model status to failed
    #         model_status_query = (
    #             update(ModelDB)
    #             .where(ModelDB.id == model_id)
    #             .values(
    #                 status="failed",
    #                 error_log=error_log,
    #                 updated_at=datetime.now(),
    #             )
    #         )
    #         result = await db.execute(model_status_query)
    #         pass
    #     else:
    #         raise ValueError(f"Unknown status: {status}")

    #     return {"message": "success"}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


class ListModelsInput(BaseModel):
    limit: int = 10
    search: Optional[str] = None


@router.get("/volume/list-models", include_in_schema=False)
async def list_models(request: Request, limit: int = 10, search: Optional[str] = None):
    api = HfApi()
    models = api.list_models(limit=limit, search=search)

    models_list = list(models)  # Convert generator to list

    print("limit: ", limit)
    print("search: ", search)

    # Define banned keywords
    banned_keywords = [".gitattributes", ".gitignore", "LICENSE.md", "README.md"]

    # Fetch detailed model info for each model
    detailed_models = []
    for model in models_list:
        try:
            model_info = await get_model_info(request, model.id)
            for sibling in model_info.siblings:
                save_path, filename = (
                    sibling.rfilename.split("/")
                    if "/" in sibling.rfilename
                    else ("", sibling.rfilename)
                )
                # Skip files with banned keywords
                if any(keyword in filename for keyword in banned_keywords):
                    continue

                converted_model = {
                    "type": model_info.pipeline_tag or "",
                    "description": model_info.card_data.get("description", ""),
                    "name": model.id,
                    "base": "",
                    "save_path": save_path,
                    "filename": filename,
                    "reference": f"https://huggingface.co/{model.id}/blob/main/{sibling.rfilename}",
                    "url": f"https://huggingface.co/{model.id}/resolve/main/{sibling.rfilename}",
                }
                # Safely extract base model information
                if model_info.tags:
                    base_tag = next(
                        (
                            tag
                            for tag in model_info.tags
                            if tag.startswith("base_model:")
                        ),
                        None,
                    )
                    if base_tag:
                        converted_model["base"] = base_tag.split(":")[-1]

                detailed_models.append(converted_model)
        except Exception as e:
            print(f"Error fetching info for model {model.id}: {str(e)}")
            continue

    return {"models": detailed_models}


@router.get("/volume/get-model-info", include_in_schema=False)
async def get_model_info(request: Request, repo_id: str):
    api = HfApi()
    model_info = api.model_info(repo_id)
    return model_info

# Helper functions for file operations
async def get_filename_from_url(url: str) -> Optional[str]:
    """Extract filename from URL or Content-Disposition header"""
    try:
        # First try to get filename from the URL path
        parsed_url = urlparse(url)
        path_filename = parsed_url.path.split('/')[-1]
        if path_filename and '.' in path_filename:
            return path_filename

        # If no filename in URL, try to get it from Content-Disposition header
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                if response.status == 200:
                    content_disposition = response.headers.get('Content-Disposition')
                    if content_disposition:
                        filename_match = re.search(r'filename="?([^"]+)"?', content_disposition)
                        if filename_match:
                            return filename_match.group(1)
        
        return None
    except Exception as e:
        logger.error(f"Error getting filename from URL: {str(e)}")
        return None

async def check_file_existence(
    filename: str, 
    custom_path: str, 
    url: str, 
    upload_type: str,
    request: Request,
    db: AsyncSession
) -> None:
    """Check if a file already exists in the volume"""
    try:
        volumes = await retrieve_model_volumes(request, db)
        volume = Volume.from_name(volumes[0]["volume_name"])
        
        full_path = os.path.join(custom_path, filename)
        try:
            contents = volume.listdir(full_path)
            if contents:
                await create_model_error_record(
                    request=request,
                    db=db,
                    url=url,
                    error_message="File already exists",
                    upload_type=upload_type,
                    folder_path=custom_path,
                    model_name=filename
                )
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {full_path} already exists"
                )
        except grpclib.exceptions.GRPCError as e:
            if e.status != grpclib.Status.NOT_FOUND:
                raise e
    except Exception as e:
        logger.error(f"Error checking file existence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Civitai handler
async def handle_civitai_model(
    request: Request,
    data: AddFileInputNew,
    db: AsyncSession,
    background_tasks: BackgroundTasks
):
    """Handle Civitai model downloads"""
    try:
        # Parse Civitai URL and get model info
        model_id, version_id = parse_civitai_url(data.url)
        civitai_info = await get_civitai_model_info(model_id)
        
        if not civitai_info.get("modelVersions"):
            raise HTTPException(status_code=400, detail="No model versions found")
            
        # Select version (latest if not specified)
        selected_version = next(
            (v for v in civitai_info["modelVersions"] 
             if str(v["id"]) == version_id) if version_id 
            else civitai_info["modelVersions"][0]
        )
        
        if not selected_version:
            raise HTTPException(status_code=400, detail="Model version not found")
            
        # Get filename
        filename = data.filename or selected_version["files"][0]["name"]
        
        # Check if file exists
        await check_file_existence(
            filename, 
            data.folder_path, 
            data.url, 
            "civitai",
            request,
            db
        )
        
        # Create model record
        volumes = await retrieve_model_volumes(request, db)
        model = ModelDB(
            user_id=request.state.current_user["user_id"],
            org_id=request.state.current_user.get("org_id"),
            upload_type="civitai",
            model_name=filename,
            civitai_id=str(civitai_info["id"]),
            civitai_version_id=str(selected_version["id"]),
            civitai_url=data.url,
            civitai_download_url=selected_version["files"][0]["downloadUrl"],
            civitai_model_response=civitai_info,
            user_volume_id=volumes[0]["id"],
            model_type="custom",
            folder_path=data.folder_path,
            status="started",
            download_progress=0
        )
        
        db.add(model)
        await db.commit()
        await db.refresh(model)
        
        # Start download
        await handle_file_download(
            request=request,
            db=db,
            download_url=selected_version["files"][0]["downloadUrl"],
            folder_path=data.folder_path,
            filename=filename,
            upload_type="civitai",
            db_model_id=str(model.id),
            background_tasks=background_tasks
        )
        
        return {"message": "Civitai model download started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling Civitai model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# HuggingFace handler
async def handle_huggingface_model(
    request: Request,
    data: AddFileInputNew,
    db: AsyncSession,
    background_tasks: BackgroundTasks
):
    """Handle HuggingFace model downloads"""
    try:
        # Extract repo ID from URL
        repo_id = extract_huggingface_repo_id(data.url)
        if not repo_id:
            raise HTTPException(status_code=400, detail="Invalid Hugging Face URL")
            
        # Get filename
        filename = data.filename or await get_filename_from_url(data.url)
        if not filename:
            await create_model_error_record(
                request=request,
                db=db,
                url=data.url,
                error_message="filename not found",
                upload_type="huggingface",
                folder_path=data.folder_path
            )
            raise HTTPException(status_code=400, detail="No filename found")
            
        # Check if file exists
        await check_file_existence(
            filename, 
            data.folder_path, 
            data.url, 
            "huggingface",
            request,
            db
        )
        
        # Create model record
        volumes = await retrieve_model_volumes(request, db)
        model = await add_model_download_url(
            request=request,
            db=db,
            upload_type="huggingface",
            model_name=filename,
            url=data.url,
            volume_id=volumes[0]["id"],
            custom_path=data.folder_path
        )
        
        # Start download
        await handle_file_download(
            request=request,
            db=db,
            download_url=data.url,
            folder_path=data.folder_path,
            filename=filename,
            upload_type="huggingface",
            db_model_id=str(model.id),
            background_tasks=background_tasks
        )
        
        return {"message": "Hugging Face model download started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling HuggingFace model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions for Civitai and HuggingFace
def parse_civitai_url(url: str) -> Tuple[str, Optional[str]]:
    """Extract model ID and version ID from Civitai URL"""
    model_match = re.search(r'civitai\.com/models/(\d+)(?:/.*?)?(?:\?modelVersionId=(\d+))?', url)
    if not model_match:
        raise HTTPException(status_code=400, detail="Invalid Civitai URL")
    return model_match.group(1), model_match.group(2)

async def get_civitai_model_info(model_id: str) -> dict:
    """Fetch model information from Civitai API"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://civitai.com/api/v1/models/{model_id}") as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail="Error fetching model info from Civitai"
                )
            return await response.json()

def extract_huggingface_repo_id(url: str) -> Optional[str]:
    """Extract repository ID from HuggingFace URL"""
    match = re.search(r'huggingface\.co/([^/]+/[^/]+)', url)
    return match.group(1) if match else None
