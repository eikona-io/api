import asyncio
import json
import os

from .workflows import CustomJSONEncoder
from .utils import ensure_run_timeout, get_user_settings, post_process_output_data
from .types import (
    WorkflowRunModel,
    WorkflowVersionModel,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from .utils import post_process_outputs, select
from sqlalchemy import func
from datetime import datetime, timedelta, timezone
from fastapi.responses import JSONResponse
from pprint import pprint

from sqlalchemy import text

import httpx
from functools import lru_cache, wraps

def async_lru_cache(maxsize=128, typed=False):
    def decorator(async_func):
        sync_func = lru_cache(maxsize=maxsize, typed=typed)(lambda *args, **kwargs: None)
        
        @wraps(async_func)
        async def wrapper(*args, **kwargs):
            cache_key = args + tuple(sorted(kwargs.items()))
            cached_result = sync_func(*cache_key)
            if cached_result is not None:
                return cached_result
            result = await async_func(*args, **kwargs)
            sync_func(*cache_key, result)
            return result
        
        return wrapper
    return decorator

from api.models import (
    Workflow,
    WorkflowRun,
    WorkflowRunWithExtra,
    WorkflowVersion,
)
from api.database import get_db
import logging
from typing import List, Optional
# from fastapi_pagination import Page, add_pagination, paginate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Workflow"])

# Add this cache at the module level
user_icon_cache = {}

@async_lru_cache(maxsize=1000)
async def fetch_user_icon(user_id: str) -> tuple[str, Optional[str]]:
    current_time = datetime.now()

    # Check if the user_id is in the cache and not expired (1 day)
    if user_id in user_icon_cache:
        cached_data, timestamp = user_icon_cache[user_id]
        if current_time - timestamp < timedelta(days=1):
            return user_id, cached_data

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.clerk.com/v1/users/{user_id}",
            headers={"Authorization": f"Bearer {clerk_token}"},
        )
        if response.status_code == 200:
            user_data = response.json()
            image_url = user_data.get("image_url")
            # Update the cache
            user_icon_cache[user_id] = (image_url, current_time)
            return user_id, image_url

    # If fetching fails, cache None for 1 hour to avoid frequent retries
    user_icon_cache[user_id] = (None, current_time)
    return user_id, None


@router.get("/workflow/{workflow_id}/runs", response_model=List[WorkflowRunModel])
async def get_all_runs(
    request: Request,
    workflow_id: str,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    user_settings = await get_user_settings(request, db)

    query = (
        select(WorkflowRunWithExtra)
        .options(
            joinedload(WorkflowRun.outputs),
            joinedload(WorkflowRun.workflow),
            joinedload(WorkflowRun.version)
        )
        .outerjoin(WorkflowVersion, WorkflowRun.workflow_version_id == WorkflowVersion.id)
        .where(WorkflowRun.workflow_id == workflow_id)
        .apply_org_check(request)
        .order_by(WorkflowRun.created_at.desc())
        .paginate(limit, offset)
    )

    result = await db.execute(query)
    runs = result.unique().scalars().all()

    if not runs:
        return []
    for run in runs:
        ensure_run_timeout(run)

    # Loop through each run and check its outputs
    for run in runs:
        if run.outputs:
            post_process_outputs(run.outputs, user_settings)

    runs_data = []
    for run in runs:
        run_dict = run.to_dict()
        if run.version:
            run_dict['version'] = run.version.to_dict()
        else:
            run_dict['version'] = None  # Explicitly set to None if no version
        runs_data.append(run_dict)

    return JSONResponse(content=runs_data)


clerk_token = os.getenv("CLERK_SECRET_KEY")


@router.get(
    "/workflow/{workflow_id}/versions", response_model=List[WorkflowVersionModel]
)
async def get_versions(
    request: Request,
    workflow_id: str,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    # Check if the user can access this workflow
    workflow = await db.execute(
        select(Workflow).where(Workflow.id == workflow_id).apply_org_check(request)
    )
    workflow = workflow.scalar_one_or_none()

    if not workflow:
        raise HTTPException(
            status_code=404, detail="Workflow not found or you don't have access to it"
        )

    query = (
        select(WorkflowVersion)
        .where(WorkflowVersion.workflow_id == workflow_id)
        .order_by(WorkflowVersion.created_at.desc())
        .paginate(limit, offset)
    )

    if search:
        query = query.where(
            func.lower(WorkflowVersion.comment).contains(search.lower())
        )

    result = await db.execute(query)
    runs = result.unique().scalars().all()

    unique_user_ids = list(set(run.user_id for run in runs if run.user_id))

    # Fetch user icons using the cached function
    results = await asyncio.gather(
        *[fetch_user_icon(user_id) for user_id in unique_user_ids]
    )

    # Process results
    user_icons = dict(results)

    if not runs:
        return []

    runs_data = [
        {**run.to_dict(), "user_icon": user_icons.get(run.user_id)} for run in runs
    ]

    return JSONResponse(content=runs_data)


@router.get("/workflow/{workflow_id}/gallery")
async def get_workflows_gallery(
    request: Request,
    workflow_id: str,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    raw_query = text("""
    SELECT
        output.id as output_id,
        run.id as run_id,
        output.data,
        output.node_meta,
        (EXTRACT(EPOCH FROM run.ended_at) - EXTRACT(EPOCH FROM run.started_at)) AS run_duration,
        (EXTRACT(EPOCH FROM run.started_at) - EXTRACT(EPOCH FROM run.queued_at)) AS cold_start,
        (EXTRACT(EPOCH FROM run.queued_at) - EXTRACT(EPOCH FROM run.created_at)) AS queue_time
    FROM
        comfyui_deploy.workflow_runs AS run
        INNER JOIN comfyui_deploy.workflow_run_outputs AS output ON run.id = output.run_id
    WHERE
        run.workflow_id = :workflow_id
        AND run.status = 'success'
        AND (output.data ?| ARRAY['images', 'gifs', 'mesh'])
        AND (
            (CAST(:org_id AS TEXT) IS NOT NULL AND run.org_id = CAST(:org_id AS TEXT))
            OR (CAST(:org_id AS TEXT) IS NULL AND run.org_id IS NULL AND run.user_id = CAST(:user_id AS TEXT))
        )
    LIMIT :limit
    OFFSET :offset
    """)

    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if 'org_id' in current_user else None

    # Execute the query
    result = await db.execute(
        raw_query,
        {
            "workflow_id": workflow_id,
            "limit": limit,
            "offset": offset,
            "org_id": org_id,
            "user_id": user_id,
        },
    )

    outputs = [
        dict(row._mapping)
        for row in result.fetchall()
    ]
    
    user_settings = await get_user_settings(request, db)
    for output in outputs:
        if output["data"]:
            post_process_output_data(output["data"], user_settings)

    # Use the custom encoder to serialize the data
    return JSONResponse(
        status_code=200, 
        content=json.loads(json.dumps(outputs, cls=CustomJSONEncoder))
    )
