import random
from uuid import UUID
from fastapi import HTTPException, APIRouter, Request
from typing import Any, Dict
import logging
import os
import httpx
from .utils import async_lru_cache
from datetime import timedelta
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool
from pydantic_ai.messages import ModelMessage

import json
from typing import Optional, Dict
from dataclasses import dataclass

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


# Define your dependencies data class - now we store workflow_json but don't automatically include it
@dataclass
class ComfyDependencies:
    workflow_json: Optional[str] = None


# Create your agent with the dependencies type
agent = Agent(
    "google-gla:gemini-2.0-flash",
    # "o3-mini",
    deps_type=ComfyDependencies,
    system_prompt="""You are Master Comfy, a specialized assistant for ComfyUI.
Focus exclusively on ComfyUI-related topics.

IMPORTANT: Never ask for user's workflow json, always run the `get_workflow_json` tool to get the workflow json, if user ask for bug fixes or troubleshooting.

**Workflow JSON Handling:**
- Always check and search for the workflow JSON when addressing issues, bug fixes, or optimization inquiries.


When troubleshooting:
- Identify specific nodes causing issues
- Provide clear, actionable solutions
- Reference relevant documentation when helpful

For workflow analysis:
- Suggest optimizations for efficiency
- Explain unusual node combinations or configurations
- Point out common mistakes in setups

When referring to specific nodes in a workflow:
- Use the special syntax `[[node:NODE_ID:NODE_POSITION]]` to reference them
- Example: "`[[node:5:342.1,-443.2]]`"
- Always include both the node ID and node position when available

Keep responses concise and practical.
Format all responses in Markdown for better readability.
Use code blocks for workflow JSON or node configurations.""",
)


# Replace the system prompt function with a function tool
@agent.tool
async def get_workflow_json(ctx: RunContext[ComfyDependencies]) -> str:
    """
    Get the workflow JSON.
    """
    print("TOOL CALLED: get_workflow_json")
    print(f"ctx: {ctx.deps}")
    if ctx.deps and ctx.deps.workflow_json:
        return ctx.deps.workflow_json
    return "No workflow JSON available. "


async def test_ai_stream():
    # Add a 1-second delay before starting to stream
    await asyncio.sleep(1)

    # Dummy response strings
    dummy_response = """OK, I can help you with that. The error message indicates that the `EmptyLatentImage` node has an invalid `batch_size` value. The minimum value allowed for `batch_size` is 1, but you\'ve set it to -1.

Here\'s how to fix it step-by-step:

1.  **Locate the `EmptyLatentImage` node:** In your ComfyUI workflow, find the node `[[node:5:342.1,-443.2]]` labeled "EmptyLatentImage".
2.  **Access the `batch_size` Widget:** Double-click the `EmptyLatentImage` node to open its properties, or simply look at the widgets displayed below the node. You should see widgets for `width`, `height`, and `batch_size`.
3.  **Change the `batch_size` value:**  The `batch_size` widget currently has the value "-1". Change this value to "1" or any positive integer greater than 1, depending on how many images you want to generate in a batch. A `batch_size` of 1 will generate images one at a time.

After making this change, try running your workflow again. The error should be resolved.
"""
    # Instead of splitting by words only, we'll first split by lines to preserve newlines
    lines = dummy_response.splitlines(True)  # Keep the newline characters

    # Process each line separately
    for line in lines:
        words = line.split()
        for word in words:
            yield f"data: {json.dumps({'text': word + ' '})}\n\n"
            # await asyncio.sleep(random.uniform(0, 0.2))

        # Add a newline after each line (if it wasn't the last line)
        if line.strip():  # Only send newline if the line wasn't empty
            yield f"data: {json.dumps({'text': '\n', 'newline': True})}\n\n"
            # await asyncio.sleep(random.uniform(0, 0.1))

    # Signal completion at the end
    yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"


# Add a dictionary to store message histories by session ID
chat_sessions: Dict[UUID, list[ModelMessage]] = {}

# Update your AiRequest model as before
class AiRequest(BaseModel):
    message: str
    chat_session_id: UUID
    is_testing: Optional[bool] = False
    workflow_json: Optional[str] = None


# Update your endpoint
@router.post("/ai")
async def ai_stream(body: AiRequest):
    """Stream AI responses word by word"""

    async def generate():
        if body.is_testing:
            async for chunk in test_ai_stream():
                yield chunk
        else:
            # Create dependencies instance with the workflow JSON
            deps = ComfyDependencies(workflow_json=body.workflow_json)

            # Get existing message history or default to empty list
            message_history = chat_sessions.get(body.chat_session_id, [])

            # Use an async context manager for run_stream with deps and message history
            async with agent.run_stream(
                body.message,
                deps=deps,
                message_history=message_history
            ) as result:
                # Stream text as it's generated
                async for message in result.stream_text(delta=True):
                    yield f"data: {json.dumps({'text': message})}\n\n"

                # After streaming is complete, save the updated message history
                # This includes both previous messages and new messages from this run
                chat_sessions[body.chat_session_id] = result.all_messages()

                # Signal completion at the end
                yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
