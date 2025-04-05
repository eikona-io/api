from fastapi import HTTPException, APIRouter, Request
from typing import Any, Dict
import logging
import os
import httpx
from .utils import async_lru_cache
from datetime import timedelta
from fastapi.responses import JSONResponse

from typing import Dict

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

