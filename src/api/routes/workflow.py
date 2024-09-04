from .types import (
    WorkflowRunModel,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from .utils import select
from sqlalchemy import func
from datetime import datetime, timedelta, timezone

# from sqlalchemy import select
from api.models import (
    WorkflowRun,
)
from api.database import get_db
import logging
from typing import List
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
    query = (
        select(WorkflowRun)
        .options(joinedload(WorkflowRun.outputs))
        .where(WorkflowRun.workflow_id == workflow_id)
        .apply_org_check(request)
        .order_by(WorkflowRun.created_at.desc())
        .paginate(limit, offset)
    )

    result = await db.execute(query)
    runs = result.unique().scalars().all()

    if not runs:
        raise HTTPException(status_code=404, detail="Runs not found")

    # Apply timeout logic
    timeout_minutes = 15
    timeout_delta = timedelta(minutes=timeout_minutes)
    now = datetime.now(timezone.utc)

    for run in runs:
        # Not started for 15 mins
        if run.status == "not-started" and now - run.created_at.replace(tzinfo=timezone.utc) > timeout_delta:
            run.status = "timeout"
        
        # Queued for 15 mins
        elif run.status == "queued" and run.queued_at and now - run.queued_at.replace(tzinfo=timezone.utc) > timeout_delta:
            run.status = "timeout"
        
        # Started for 15 mins
        elif run.status == "started" and run.started_at and now - run.started_at.replace(tzinfo=timezone.utc) > timeout_delta:
            run.status = "timeout"
        
        # Running and not updated in the last 15 mins
        elif run.status not in ["success", "failed", "timeout", "cancelled"]:
            updated_at = run.updated_at.replace(tzinfo=timezone.utc) if run.updated_at.tzinfo is None else run.updated_at
            if now - updated_at > timeout_delta:
                run.status = "timeout"

    return runs
