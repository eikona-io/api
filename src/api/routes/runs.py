import logging
from typing import List, Optional
from .types import WorkflowRunModel
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from datetime import datetime
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


def clean_input(value: Optional[str]) -> Optional[str]:
    return value if value and value.strip() else None


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
    # filter origin
    origin: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    try:
        # Convert string to datetime
        start_time = (
            datetime.fromtimestamp(max(1, start_time_unix)) if start_time_unix else None
        )
        end_time = (
            datetime.fromtimestamp(max(1, end_time_unix)) if end_time_unix else None
        )

        current_user = request.state.current_user
        user_id = current_user["user_id"]
        org_id = current_user["org_id"] if "org_id" in current_user else None

        # Clean input parameters
        workflow_id = clean_input(workflow_id)
        status = clean_input(status)
        gpu = clean_input(gpu)
        machine_id = clean_input(machine_id)
        origin = clean_input(origin)

        conditions = [
            ("created_at >= :start_time", start_time),
            ("created_at <= :end_time", end_time),
            ("org_id = :org_id", org_id),
            ("user_id = :user_id", user_id),
            ("workflow_id = :workflow_id", workflow_id),
            ("status = :status", status),
            ("gpu = :gpu", gpu),
            ("machine_id = :machine_id", machine_id),
            ("origin = :origin", origin),
        ]

        where_clause = " AND ".join(cond for cond, val in conditions if val is not None)

        query = text(f"""
    WITH filtered_workflow_runs AS (
      SELECT
        id, status, created_at, gpu, workflow_id, machine_id, origin
      FROM "comfyui_deploy"."workflow_runs"
      WHERE {where_clause}
      ORDER BY created_at DESC
    )
    SELECT 
      filtered_workflow_runs.created_at,
      filtered_workflow_runs.id AS run_id,
      filtered_workflow_runs.gpu,
      filtered_workflow_runs.origin,
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
            key: val
            for key, val in {
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
                "origin": origin,
            }.items()
            if val is not None
        }

        result = await db.execute(query, params)
        runs = result.fetchall()

        runs_data = [
            {
                "created_at": run.created_at,
                "id": run.run_id,
                "gpu": run.gpu,
                "origin": run.origin,
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
    except Exception as e:
        logger.error(f"Error getting runs: {e}")
        return JSONResponse(status_code=500, content={"Error. "})
