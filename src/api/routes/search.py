from fastapi import APIRouter
from typing import List, Optional
from huggingface_hub import HfApi
from pydantic import BaseModel
from datetime import timedelta
from .utils import async_lru_cache
import asyncio
from huggingface_hub import run_as_future
from huggingface_hub import ModelInfo
import time
import fnmatch
import aiohttp
import os
from typing import Dict, Any
import json

router = APIRouter(tags=["Search"])


class ModelSearchQuery(BaseModel):
    name: str = None
    type: Optional[str] = None
    provider: str = None
    filename: str = None
    save_path: str = None
    size: Optional[int] = None
    download_url: str = None
    reference_url: str = None


class SearchModelsResponse(BaseModel):
    models: List[ModelSearchQuery]


_hf_api = None


def get_hf_api():
    global _hf_api
    if _hf_api is None:
        _hf_api = HfApi()
    return _hf_api


@router.get("/search/model", response_model=SearchModelsResponse)
async def search(query: str, provider: str = "all"):
    results = []
    tasks = []

    if provider in ["all", "comfyui"]:
        tasks.append(search_comfyui(query))
        
    if provider in ["all", "civitai"]:
        tasks.append(search_civitai(query))
        
    if provider in ["all", "huggingface"]:
        hf_api = get_hf_api()
        tasks.append(search_hf(query, hf_api))

    # Run all search tasks in parallel
    search_results = await asyncio.gather(*tasks)

    # Combine results from all searches
    for result_list in search_results:
        results.extend(result_list)

    return SearchModelsResponse(models=results)


limit = 5

# @async_lru_cache(maxsize=200, expire_after=timedelta(hours=1))
# async def get_model_info(repo_id: str):
#     return await run_as_future(hf_api.model_info, repo_id)


@async_lru_cache(maxsize=200, expire_after=timedelta(hours=1))
async def search_hf(search: str, hf_api: HfApi):
    # Create the future first
    future = run_as_future(
        hf_api.list_models, limit=limit, search=search, expand=["siblings"]
    )

    # Poll the future until it's done
    while not future.done():
        time.sleep(0.1)  # Sleep for a short time before checking again

    # Once done, get the result
    models_list = future.result()
    models_list: List[ModelInfo] = list(models_list)  # Convert generator to list

    # blacklisted_files = [".gitattributes", ".gitignore", "LICENSE.md", "README.md"]
    whitelisted_files = ["*.safetensors", "*.ckpt", "*.pt", "*.pth"]

    results = []
    for model_info in models_list:
        # Add siblings if they exist
        if model_info.siblings:
            for sibling in model_info.siblings:
                if any(
                    fnmatch.fnmatch(sibling.rfilename, pattern)
                    for pattern in whitelisted_files
                ):
                    results.append(
                        ModelSearchQuery(
                            name=f"{model_info.id}/{sibling.rfilename}",
                            type="",  # This will include file type
                            provider="huggingface",
                            filename=os.path.basename(sibling.rfilename),
                            size=sibling.size,
                            reference_url=f"https://huggingface.co/{model_info.id}/blob/main/{sibling.rfilename}",
                            download_url=f"https://huggingface.co/{model_info.id}/resolve/main/{sibling.rfilename}",
                        )
                    )

    return results


def map_type(type_str: str) -> str:
    type_mapping = {
        "checkpoint": "checkpoints",
        # Add more mappings as needed
    }
    return type_mapping.get(type_str, type_str)

def map_type_to_save_path(type_str: str) -> str:
    type_mapping = {
        "checkpoint": "checkpoints",
        # Add more mappings as needed
    }
    return type_mapping.get(type_str, type_str)


@async_lru_cache(maxsize=200, expire_after=timedelta(hours=1))
async def search_civitai(search: str):
    base_url = "https://civitai.com/api/v1/models"
    params = {"limit": limit, "sort": "Most Downloaded", "query": search}

    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            data = await response.json()

            results = []
            for item in data["items"]:
                for version in item["modelVersions"]:
                    for file in version["files"]:
                        # Convert size from KB to bytes, ensuring integer values
                        size_in_bytes = (
                            int(file.get("sizeKB", 0) * 1024)
                            if file.get("sizeKB") is not None
                            else None
                        )
                        # print(item)
                        results.append(
                            ModelSearchQuery(
                                name=f"{item['name']} {version['name']}",
                                type=map_type(
                                    item.get("type", "").lower()
                                ),  # Apply the mapping here
                                save_path=map_type_to_save_path(
                                    item.get("type", "").lower()
                                ),
                                provider="civitai",
                                size=size_in_bytes,
                                filename=file["name"],
                                reference_url=f"https://civitai.com/models/{item['id']}?modelVersionId={version['id']}",
                                download_url=file.get("downloadUrl", ""),
                            )
                        )

            return results


@async_lru_cache(maxsize=1, expire_after=timedelta(hours=1))
async def get_comfyui_data():
    url = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            text = await response.text()
            try:
                return await response.json()
            except:
                # If direct json() fails, try parsing the text content
                try:
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse JSON response: {e}, response: {text}"
                    )


@async_lru_cache(maxsize=200, expire_after=timedelta(hours=1))
async def search_comfyui(search: str):
    data = await get_comfyui_data()

    results = []
    for model in data.get("models", []):
        if search.lower() in model.get("name", "").lower():
            download_url = model.get("url", "")
            # Some ComfyUI models use reference_url instead of url
            if not download_url:
                download_url = model.get("reference", "")

            # Convert size from GB string to bytes
            size_str = model.get("size")
            size_bytes = None
            if size_str:
                try:
                    # Remove unit and convert to float
                    size_value = float(
                        size_str.replace("GB", "")
                        .replace("MB", "")
                        .replace("KB", "")
                        .strip()
                    )

                    # Convert to bytes based on unit
                    if "GB" in size_str:
                        size_bytes = int(size_value * 1024 * 1024 * 1024)
                    elif "MB" in size_str:
                        size_bytes = int(size_value * 1024 * 1024)
                    elif "KB" in size_str:
                        size_bytes = int(size_value * 1024)
                    else:
                        size_bytes = int(size_value)  # Assume bytes if no unit
                except (ValueError, AttributeError):
                    size_bytes = None

            results.append(
                ModelSearchQuery(
                    name=model.get("name", ""),
                    type=model.get("type", ""),
                    provider="comfyui",
                    filename=model.get("filename", ""),
                    save_path=model.get("save_path", ""),
                    size=size_bytes,  # Now using converted size in bytes
                    reference_url=model.get("reference_url", download_url),
                    download_url=download_url,
                )
            )

            if len(results) >= limit:
                break

    return results
