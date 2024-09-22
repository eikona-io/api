import logging
from fastapi import APIRouter
from typing import List, Optional
from .types import WorkflowRunModel
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.models import Workflow, WorkflowRun, Machine
from .utils import select
from datetime import datetime, timezone
from sqlalchemy.orm import joinedload
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Runs"])


@router.get("/runs", response_model=List[WorkflowRunModel])
async def get_runs(
    request: Request,
    limit: int = 100,
    offset: int = 0,
    # filter time range
    start_time_unix: int = None,
    end_time_unix: int = None,
    # filter workflow
    workflow_id: Optional[str] = None,
    # filter status
    status: Optional[str] = None,
    # filter gpu
    gpu: Optional[str] = None,
    # filter machine
    machine_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    start_time = datetime.fromtimestamp(start_time_unix, tz=timezone.utc)
    end_time = datetime.fromtimestamp(end_time_unix, tz=timezone.utc)

    print("start_time", start_time)
    print("end_time", end_time)

    query = (
        select(WorkflowRun)
        .where(WorkflowRun.created_at >= start_time)  # filter time range
        .where(WorkflowRun.created_at <= end_time)
        .apply_org_check(request)
        .options(
            joinedload(WorkflowRun.workflow).load_only(Workflow.name),
            joinedload(WorkflowRun.machine).load_only(Machine.name),
        )
        .order_by(WorkflowRun.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    if workflow_id:
        query = query.where(WorkflowRun.workflow_id == workflow_id)

    if status:
        query = query.where(WorkflowRun.status == status)

    if gpu:
        query = query.where(WorkflowRun.gpu == gpu)

    if machine_id:
        query = query.where(WorkflowRun.machine_id == machine_id)

    result = await db.execute(query)
    runs = result.scalars().all()

    runs_data = [run.to_dict() for run in runs]

    return JSONResponse(content=runs_data)
