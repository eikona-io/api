from fastapi import HTTPException, APIRouter, Request
from typing import Any, Dict
import logging
import os
import httpx
from .utils import async_lru_cache
from datetime import timedelta
from fastapi.responses import JSONResponse, FileResponse
import modal
import json
from api.utils.multi_level_cache import multi_level_cached


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Comfy Node"])


import asyncio
from urllib.parse import urlparse


async def extract_repo_name(repo_url: str) -> str:
    cleaned_url = repo_url.strip("'\"")
    url = urlparse(cleaned_url)
    path_parts = url.path.split("/")
    path_parts = [p for p in path_parts if p]
    repo_name = path_parts[1].replace(".git", "")
    author = path_parts[0]
    return f"{author}/{repo_name}"


async def fetch_github_data(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

@async_lru_cache(expire_after=timedelta(hours=1))
async def _get_branch_info(git_url: str) -> Dict[str, Any] | None:
    try:
        repo_name = await extract_repo_name(git_url)
        headers = {
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
            "User-Agent": "request",
        }

        repo_url = f"https://api.github.com/repos/{repo_name}"
        repo_data = await fetch_github_data(repo_url, headers)

        branch = repo_data.get("default_branch")
        print(f"Branch: {branch}")
        if not branch:
            return None

        branch_url = f"{repo_url}/branches/{branch}"
        branch_info = await fetch_github_data(branch_url, headers)

        branch_info["stargazers_count"] = repo_data.get("stargazers_count", 0)

        return branch_info
    except httpx.HTTPStatusError as e:
        logger.error(f"GitHub API error: {e.response.status_code} {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Error fetching repo info: {str(e)}")
        return None


@router.get("/branch-info", response_model=Dict[str, Any])
async def get_branch_info(request: Request, git_url: str):
    if not git_url:
        raise HTTPException(status_code=400, detail="Git URL is required")
    branch_info = await _get_branch_info(git_url)
    if branch_info is None:
        raise HTTPException(status_code=404, detail="Branch information not found")
    return JSONResponse(content=branch_info)

@async_lru_cache(expire_after=timedelta(hours=1))
async def _get_releases(git_url: str) -> list[Dict[str, Any]] | None:
    try:
        repo_name = await extract_repo_name(git_url)
        headers = {
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
            "User-Agent": "request",
        }

        releases_url = f"https://api.github.com/repos/{repo_name}/releases"
        releases = await fetch_github_data(releases_url, headers)
        return releases
    except httpx.HTTPStatusError as e:
        logger.error(f"GitHub API error: {e.response.status_code} {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Error fetching releases: {str(e)}")
        return None

async def _get_commit_sha_for_tag_with_headers(repo_name: str, tag: str) -> str | None:
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
        "User-Agent": "request",
    }
    try:
        # Get the commit SHA for a specific tag
        tag_url = f"https://api.github.com/repos/{repo_name}/git/refs/tags/{tag}"
        tag_data = await fetch_github_data(tag_url, headers)
        return tag_data["object"]["sha"]
    except Exception as e:
        logger.error(f"Error fetching commit SHA for tag {tag}: {str(e)}")
        return None

@multi_level_cached(key_prefix="comfyui_deploy_hash", ttl_seconds=3600, redis_ttl_seconds=21600)
async def get_latest_comfydeploy_hash():
    """Fetch the latest commit hash from ComfyUI-Deploy repository"""
    git_url = "https://github.com/bennykok/comfyui-deploy"
    branch_info = await _get_branch_info(git_url)
    return branch_info["commit"]["sha"] if branch_info else None

@router.get("/comfyui-versions", response_model=Dict[str, Any])
async def get_comfyui_versions(request: Request):
    git_url = "https://github.com/comfyanonymous/ComfyUI"
    repo_name = await extract_repo_name(git_url)
    
    # Get branch info and releases
    branch_info, releases = await asyncio.gather(
        _get_branch_info(git_url),
        _get_releases(git_url)
    )
    
    if branch_info is None:
        raise HTTPException(status_code=404, detail="Branch information not found")
    
    # Take only the 3 most recent releases
    recent_releases = releases[:3] if releases else []
    
    # Fetch commit SHAs for recent releases concurrently
    if recent_releases:
        sha_tasks = [
            _get_commit_sha_for_tag_with_headers(repo_name, release["tag_name"])
            for release in recent_releases
        ]
        commit_shas = await asyncio.gather(*sha_tasks)
    else:
        commit_shas = []
        
    response = {
        "latest": {
            "sha": branch_info["commit"]["sha"],
            "label": "Latest"
        },
        "releases": [
            {
                "label": release["tag_name"],
                "value": sha or release["target_commitish"],
                "description": release["name"],
                "date": release["published_at"]
            }
            for release, sha in zip(recent_releases, commit_shas) if sha
        ]
    }
    
    return JSONResponse(content=response)

@router.get("/latest-hashes", response_model=Dict[str, str])
async def get_latest_hashes():
    """Return latest hashes for ComfyUI and ComfyUI-Deploy"""
    comfyui_versions = await get_comfyui_versions(Request())
    comfydeploy_hash = await get_latest_comfydeploy_hash()
    
    return JSONResponse(content={
        "comfyui_hash": comfyui_versions["latest"]["sha"],
        "comfydeploy_hash": comfydeploy_hash
    })

@router.get("/custom-node-list")
async def get_nodes_json():
    try:
        # Get the data from modal function
        function = await modal.Function.lookup.aio("comfy-nodes", "read_custom_nodes")
        nodes_data = await function.remote.aio()

        # Create a temporary file to serve
        temp_file_path = "/tmp/nodes_cache.json"
        with open(temp_file_path, "w") as f:
            json.dump(nodes_data, f)

        headers = {
            "Cache-Control": "public, max-age=86400",
            "ETag": f"\"{hash(json.dumps(nodes_data))}\"",
        }

        return FileResponse(
            path=temp_file_path,
            media_type="application/json",
            filename="nodes.json",
            headers=headers
        )
    except Exception as e:
        logger.error(f"Error fetching nodes.json: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve nodes.json: {str(e)}")
