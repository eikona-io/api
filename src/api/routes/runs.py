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
from sqlalchemy import text
import json
from decimal import Decimal
from uuid import UUID

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Runs"])


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, UUID)):
            return str(obj)
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


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
    # Convert string to datetime
    start_time = (
        datetime.fromtimestamp(max(1, start_time_unix)) if start_time_unix else None
    )
    end_time = datetime.fromtimestamp(max(1, end_time_unix)) if end_time_unix else None

    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    conditions = ["created_at >= :start_time", "created_at <= :end_time"]
    if org_id:
        conditions.append("org_id = :org_id")
    if user_id:
        conditions.append("user_id = :user_id")
    if workflow_id:
        conditions.append("workflow_id = :workflow_id")
    if status:
        conditions.append("status = :status")
    if gpu:
        conditions.append("gpu = :gpu")
    if machine_id:
        conditions.append("machine_id = :machine_id")

    where_clause = " AND ".join(conditions)

    query = text(f"""
    WITH filtered_workflow_runs AS (
      SELECT
        id, status, created_at, gpu, workflow_id, machine_id
      FROM "comfyui_deploy"."workflow_runs"
      WHERE {where_clause}
      ORDER BY created_at DESC
    )
    SELECT 
      filtered_workflow_runs.created_at,
      filtered_workflow_runs.id AS run_id,
      filtered_workflow_runs.gpu,
      filtered_workflow_runs.status AS run_status,
      filtered_workflow_runs.workflow_id,
      "comfyui_deploy"."workflows".name AS workflow_name,
      filtered_workflow_runs.machine_id,
      "comfyui_deploy"."machines".name AS machine_name
    FROM 
      filtered_workflow_runs
    LEFT JOIN 
      "comfyui_deploy"."workflows" ON filtered_workflow_runs.workflow_id = "comfyui_deploy"."workflows".id
    LEFT JOIN 
      "comfyui_deploy"."machines" ON filtered_workflow_runs.machine_id = "comfyui_deploy"."machines".id
    LIMIT :limit OFFSET :offset;
    """)

    params = {
        "start_time": start_time,
        "end_time": end_time,
        "org_id": org_id,
        "user_id": user_id,
        "limit": limit,
        "offset": offset,
        "workflow_id": workflow_id,
        "status": status,
        "gpu": gpu,
        "machine_id": machine_id,
    }

    result = await db.execute(query, params)
    runs = result.fetchall()

    runs_data = [
        {
            "created_at": run.created_at,
            "id": run.run_id,
            "gpu": run.gpu,
            "status": run.run_status,
            "workflow": {"id": run.workflow_id, "name": run.workflow_name},
            "machine": {"id": run.machine_id, "name": run.machine_name},
        }
        for run in runs
    ]

    return JSONResponse(
        status_code=200,
        content=json.loads(json.dumps(runs_data, cls=CustomJSONEncoder)),
    )
