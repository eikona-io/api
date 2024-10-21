import time
from .types import VolFSStructure, Model
from fastapi import HTTPException, APIRouter, Request, BackgroundTasks
from typing import Any, Dict, List, Tuple, Union, Optional
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
from pydantic import BaseModel
from enum import Enum
import re
import grpclib
from modal import Volume
import modal
from huggingface_hub import HfApi

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


async def get_volume_list(request: Request, volume_name: str) -> VolFSStructure:
    if not volume_name:
        raise ValueError("Volume name is not provided")
    endpoint = f"{os.environ.get('CURRENT_API_URL')}/api/volume/ls_full?volume_name={volume_name}&create_if_missing=true"

    # Get the auth token from the request
    auth_token = request.headers.get("Authorization")
    if not auth_token:
        raise HTTPException(status_code=401, detail="Authorization token is missing")

    headers = {"Cache-Control": "no-cache", "Authorization": auth_token}

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.get(endpoint, headers=headers)
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
            model.folder_path + "/" + model.model_name for model in existing_models
        }

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
                    logger.info(f"existing_model_names {existing_model_names}")
                    logger.info(
                        f"item.path, {item.path}, {item.path not in existing_model_names}"
                    )
                    if item.path not in existing_model_names:
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
        current_folder.contents.append(VolFile(path=file_path, type="file"))

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
    volume_name: str
    download_url: str
    folder_path: str
    filename: str
    callback_url: str
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


@router.post("/volume/rename_file")
async def rename_file(request: Request, body: RenameFileBody):
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


@router.post("/volume/rm")
async def remove_file(request: Request, body: RemoveFileInput):
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


async def download_file_task(
    download_url,
    folder_path,
    filename,
    callback_url,
    db_model_id,
    full_path,
    volume_name,
    upload_type,
):
    app = modal.App("volume-operations")

    @app.function(
        serialized=True,
        image=modal.Image.debian_slim().pip_install("aiohttp", "huggingface_hub"),
    )
    async def download_file_task(
        download_url,
        folder_path,
        filename,
        callback_url,
        db_model_id,
        full_path,
        volume_name,
        upload_type,
        token,
    ):
        import time
        import os
        from enum import Enum
        from modal import Volume
        import aiohttp
        from huggingface_hub import hf_hub_download, HfApi
        import re

        print("download_file_task start")
        print("callback_url", callback_url)

        class ModelDownloadStatus(Enum):
            PROGRESS = "progress"
            SUCCESS = "success"
            FAILED = "failed"

        async def progress_callback(
            callback_url,
            model_id,
            progress,
            status: ModelDownloadStatus,
        ):
            payload = {
                "model_id": model_id,
                "download_progress": progress,
                "status": status.value,
            }
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        callback_url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                        },
                    )
            except Exception as e:
                print("error in progress callback: ", e)

        def extract_huggingface_repo_id(url: str) -> str | None:
            match = re.search(r"huggingface\.co/([^/]+/[^/]+)", url)
            return match.group(1) if match else None

        async def download_url_file(download_url):
            headers = {"Accept-Encoding": "identity"}
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(download_url, headers=headers) as response:
                        response.raise_for_status()
                        total_size = int(response.headers.get("Content-Length", 0))
                        downloaded_size = 0
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        last_callback_time = time.time()

                        with open(full_path, "wb") as file:
                            async for data in response.content.iter_chunked(8192):
                                if data:
                                    file.write(data)
                                    downloaded_size += len(data)
                                    progress = (
                                        int((downloaded_size / total_size) * 90)
                                        if total_size
                                        else 0
                                    )
                                    current_time = time.time()
                                    if current_time - last_callback_time >= 5:
                                        print("=======================================")
                                        print("download url started")
                                        print("folder_path: ", folder_path)
                                        print("filename: ", filename)
                                        print("full_path: ", full_path)
                                        print("downloaded_size: ", downloaded_size)
                                        print("total_size: ", total_size)
                                        print("progress: ", progress)
                                        print("=======================================")
                                        await progress_callback(
                                            callback_url,
                                            db_model_id,
                                            progress,
                                            ModelDownloadStatus.PROGRESS,
                                        )
                                        last_callback_time = current_time

                return full_path
            except Exception as e:
                await progress_callback(
                    callback_url, db_model_id, 0, ModelDownloadStatus.FAILED
                )
                raise e

        async def download_hf_model(folder_path):
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                # def custom_progress_callback(progress):
                #     print(f"Download progress: {progress:.2f}%")
                #     progress_callback(
                #         callback_url,
                #         db_model_id,
                #         int(progress),
                #         ModelDownloadStatus.PROGRESS,
                #     )

                repo_id = extract_huggingface_repo_id(download_url)

                print("=======================================")
                print("huggingface started")
                print("repo_id: ", repo_id)
                print("folder_path: ", folder_path)
                print("filename: ", filename)
                print("full_path: ", full_path)
                print("=======================================")

                folder_path = folder_path.rstrip("/")

                downloaded_model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=folder_path,
                    force_download=True,
                    token=token,
                )

                return downloaded_model_path
            except Exception as e:
                await progress_callback(
                    callback_url, db_model_id, 0, ModelDownloadStatus.FAILED
                )
                raise e

        if upload_type == "huggingface":
            downloaded_path = await download_hf_model(folder_path)
        elif upload_type == "download-url":
            if "civitai.com" in download_url:
                download_url += f"{'&' if '?' in download_url else '?'}token={os.environ['CIVITAI_KEY']}"
            downloaded_path = await download_url_file(download_url)
        else:
            raise ValueError(f"Unsupported upload_type: {upload_type}")

        volume = Volume.lookup(volume_name, create_if_missing=True)
        print("=======================================")
        print("download_file_task done")
        print("download_url: ", download_url)
        print("folder_path: ", folder_path)
        print("filename: ", filename)
        print("callback_url: ", callback_url)
        print("body.db_model_id: ", db_model_id)
        print("=======================================")

        with volume.batch_upload() as batch:
            batch.put_file(downloaded_path, full_path)

        try:
            await progress_callback(
                callback_url, db_model_id, 100, ModelDownloadStatus.SUCCESS
            )
        except Exception as e:
            print(f"Failed to send progress callback: {str(e)}")

    async with app.run.aio():
        await download_file_task.remote.aio(
            download_url,
            folder_path,
            filename,
            callback_url,
            db_model_id,
            full_path,
            volume_name,
            upload_type,
            token=os.environ.get("HUGGINGFACE_TOKEN"),
        )


@router.post("/volume/add_file")
async def add_file(
    request: Request, body: AddFileInput, background_tasks: BackgroundTasks
):
    volume_name = body.volume_name
    download_url = body.download_url
    folder_path = body.folder_path
    filename = body.filename
    callback_url = body.callback_url
    upload_type = body.upload_type

    volume = lookup_volume(volume_name, create_if_missing=True)
    full_path = os.path.join(folder_path, filename)

    print("volume: ", volume)
    print("full_path: ", full_path)

    try:
        file_exists = does_file_exist(full_path, volume)
        if file_exists:
            raise HTTPException(status_code=400, detail="File already exists.")
    except grpclib.exceptions.GRPCError as e:
        print("e: ", str(e))
        raise HTTPException(status_code=400, detail="Error: " + str(e))

    background_tasks.add_task(
        download_file_task,
        download_url,
        folder_path,
        filename,
        callback_url,
        body.db_model_id,
        full_path,
        volume_name,
        upload_type,
    )

    return {"full_path": full_path}


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


@router.get("/volume/ls_full")
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
                # "size": content.size if content.type == FILE_TYPE else None,
                "contents": [] if content.type == DIRECTORY_TYPE else None,
            }

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


@router.get("/volume/ls")
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

class ListModelsInput(BaseModel):
    limit: int = 10
    search: Optional[str] = None


@router.get("/volume/list-models")
async def list_models(request: Request, body: ListModelsInput):
    limit = body.limit
    search = body.search

    api = HfApi()
    models = api.list_models(limit=limit, search=search)

    models_list = list(models)  # Convert generator to list

    if models_list:
        model = models_list[0]
        print("=======================================")
        print("Sample model details:")
        print(f"Model ID: {model.id}")
        print(f"Last modified: {model.last_modified}")
        print(f"Tags: {model.tags}")
        print("=======================================")
        return models_list
        # return {
        #     "id": model.id,
        #     "last_modified": model.last_modified,
        #     "tags": model.tags,
        # }
    else:
        return {"message": "No models found"}


@router.get("/volume/get-model-info")
async def get_model_info(request: Request, repo_id: str):
    api = HfApi()
    model_info = api.model_info(repo_id)
    return model_info
