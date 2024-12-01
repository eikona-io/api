import asyncio
import json
import os

from .workflows import CustomJSONEncoder
from .utils import (
    UserIconData,
    ensure_run_timeout,
    fetch_user_icon,
    get_user_settings,
    post_process_output_data,
)
from .types import (
    WorkflowModel,
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


from api.models import (
    Deployment,
    Machine,
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
            joinedload(WorkflowRun.version),
        )
        .outerjoin(
            WorkflowVersion, WorkflowRun.workflow_version_id == WorkflowVersion.id
        )
        .join(Workflow, WorkflowRun.workflow_id == Workflow.id)
        .where(WorkflowRun.workflow_id == workflow_id)
        .where(Workflow.deleted == False)
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
            run_dict["version"] = run.version.to_dict()
        else:
            run_dict["version"] = None  # Explicitly set to None if no version
        runs_data.append(run_dict)

    return JSONResponse(content=runs_data)


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
        .join(Workflow, WorkflowVersion.workflow_id == Workflow.id)
        .where(WorkflowVersion.workflow_id == workflow_id)
        .where(Workflow.deleted == False)
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

    # Process results - create dictionary by pairing user IDs with their icon data
    user_icons = {str(user_id): icon_data for user_id, icon_data in zip(unique_user_ids, results)}

    if not runs:
        return []

    runs_data = [
        {**run.to_dict(), "user_icon": user_icons.get(run.user_id).image_url if user_icons.get(run.user_id) else None}
        for run in runs
    ]

    return JSONResponse(content=runs_data)


from sqlalchemy import text


@router.get("/workflow/{workflow_id}", response_model=WorkflowModel)
async def get_workflow(
    request: Request,
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    limit: int = 1,  # Default to 5 most recent versions
):
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user.get("org_id")

    query = text("""
    WITH recent_versions AS (
        SELECT *
        FROM comfyui_deploy.workflow_versions
        WHERE workflow_id = :workflow_id
        ORDER BY created_at DESC
        LIMIT :limit
    )
    SELECT w.*, 
           json_agg(rv.* ORDER BY rv.created_at DESC) AS versions
    FROM comfyui_deploy.workflows w
    LEFT JOIN recent_versions rv ON w.id = rv.workflow_id
    WHERE w.id = :workflow_id
    AND w.deleted = false
    AND (
        (CAST(:org_id AS TEXT) IS NOT NULL AND w.org_id = CAST(:org_id AS TEXT))
        OR (CAST(:org_id AS TEXT) IS NULL AND w.org_id IS NULL AND w.user_id = CAST(:user_id AS TEXT))
    )
    GROUP BY w.id
    """)

    result = await db.execute(
        query,
        {
            "workflow_id": workflow_id,
            "limit": limit,
            "org_id": org_id,
            "user_id": user_id,
        },
    )

    workflow = result.fetchone()

    if not workflow:
        raise HTTPException(
            status_code=404, detail="Workflow not found or you don't have access to it"
        )

    # Convert the result to a dict
    workflow_dict = dict(workflow._mapping)
    # Parse the JSON string of versions back into a list of dicts
    # workflow_dict['versions'] = json.loads(workflow_dict['versions'])

    return JSONResponse(
        status_code=200,
        content=json.loads(json.dumps(workflow_dict, cls=CustomJSONEncoder)),
    )


@router.get("/workflow/{workflow_id}/version/{version}", response_model=WorkflowModel)
async def get_workflow_version(
    request: Request,
    workflow_id: str,
    version: int,
    db: AsyncSession = Depends(get_db),
):
    workflow_version = await db.execute(
        select(WorkflowVersion)
        .join(Workflow, Workflow.id == WorkflowVersion.workflow_id)
        .where(WorkflowVersion.workflow_id == workflow_id)
        .where(WorkflowVersion.version == version)
        .where(Workflow.deleted == False)
        .apply_org_check_by_type(Workflow, request)
    )
    workflow_version = workflow_version.scalar_one_or_none()

    if not workflow_version:
        raise HTTPException(
            status_code=404,
            detail="Workflow version not found or you don't have access to it",
        )

    return JSONResponse(content=workflow_version.to_dict())


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
        INNER JOIN comfyui_deploy.workflows AS workflow ON run.workflow_id = workflow.id
    WHERE
        run.workflow_id = :workflow_id
        AND run.status = 'success'
        AND workflow.deleted = false
        AND (output.data ?| ARRAY['images', 'gifs', 'mesh'])
        AND (
            (CAST(:org_id AS TEXT) IS NOT NULL AND run.org_id = CAST(:org_id AS TEXT))
            OR (CAST(:org_id AS TEXT) IS NULL AND run.org_id IS NULL AND run.user_id = CAST(:user_id AS TEXT))
        )
    ORDER BY output.created_at desc
    LIMIT :limit
    OFFSET :offset
    """)

    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

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

    outputs = [dict(row._mapping) for row in result.fetchall()]

    user_settings = await get_user_settings(request, db)
    for output in outputs:
        if output["data"]:
            post_process_output_data(output["data"], user_settings)

    # Use the custom encoder to serialize the data
    return JSONResponse(
        status_code=200, content=json.loads(json.dumps(outputs, cls=CustomJSONEncoder))
    )


@router.get("/workflow/{workflow_id}/deployments", response_model=WorkflowModel)
async def get_deployments(
    request: Request,
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
):
    deployments = await db.execute(
        select(Deployment)
        .options(
            joinedload(Deployment.machine).load_only(Machine.name, Machine.id),
            joinedload(Deployment.version),
        )
        .join(Workflow, Workflow.id == Deployment.workflow_id)
        .where(Deployment.workflow_id == workflow_id)
        .where(Workflow.deleted == False)
        .apply_org_check(request)
        .order_by(Deployment.environment.desc())
    )
    deployments = deployments.scalars().all()

    if not deployments:
        raise HTTPException(
            status_code=404,
            detail="Deployments not found or you don't have access to it",
        )

    deployments_data = [deployment.to_dict() for deployment in deployments]

    return JSONResponse(content=deployments_data)
