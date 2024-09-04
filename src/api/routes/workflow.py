import os
from .utils import ensure_run_timeout, get_user_settings
from .types import (
    WorkflowRunModel,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from .utils import post_process_outputs, select
from sqlalchemy import func
from datetime import datetime, timedelta, timezone
from fastapi.responses import JSONResponse
from pprint import pprint

# from sqlalchemy import select
from api.models import (
    WorkflowRun,
    WorkflowRunWithExtra,
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
    user_settings = await get_user_settings(request, db)
    
    query = (
        select(WorkflowRunWithExtra)
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

    for run in runs:
        ensure_run_timeout(run)
        

    # Loop through each run and check its outputs
    for run in runs:
        if run.outputs:
            post_process_outputs(run.outputs, user_settings)

    # return runs

    runs_data = [run.to_dict() for run in runs]

    return JSONResponse(content=runs_data)