import logging
from typing import List, Optional
from .types import WorkflowRunModel
from fastapi import APIRouter, Depends, Request, Query
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
    workflow_id: Optional[List[str]] = Query(None),
    # filter status
    status: Optional[List[str]] = Query(None),
    # filter gpu
    gpu: Optional[List[str]] = Query(None),
    # filter machine
    machine_id: Optional[List[str]] = Query(None),
    # filter origin
    origin: Optional[List[str]] = Query(None),
    # filter workflow version
    workflow_version_id: Optional[List[str]] = Query(None),
    # filter queued duration
    min_queued_duration: Optional[str] = 0,
    # filter run duration
    min_run_duration: Optional[str] = 0,
    # filter total upload duration
    min_total_upload_duration: Optional[str] = 0,
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
        workflow_id = [clean_input(c) for c in workflow_id] if workflow_id else None
        status = [clean_input(s) for s in status] if status else None
        gpu = [clean_input(g) for g in gpu] if gpu else None
        machine_id = [clean_input(m) for m in machine_id] if machine_id else None
        origin = [clean_input(o) for o in origin] if origin else None
        workflow_version_id = (
            [clean_input(w) for w in workflow_version_id]
            if workflow_version_id
            else None
        )

        if min_queued_duration == "":
            min_queued_duration = 0
        if min_run_duration == "":
            min_run_duration = 0
        if min_total_upload_duration == "":
            min_total_upload_duration = 0

        conditions = [
            ("created_at >= :start_time", start_time),
            ("created_at <= :end_time", end_time),
            ("workflow_id = ANY(:workflow_id)", workflow_id),
            ("status = ANY(:status)", status),
            ("gpu = ANY(:gpu)", gpu),
            ("machine_id = ANY(:machine_id)", machine_id),
            ("origin = ANY(:origin)", origin),
            ("workflow_version_id = ANY(:workflow_version_id)", workflow_version_id),
        ]

        # Special handling for org_id and user_id
        if org_id:
            conditions.append(("org_id = :org_id", org_id))
        else:
            conditions.extend([
                ("org_id IS NULL", True),
                ("user_id = :user_id", user_id)
            ])

        where_clause = " AND ".join(cond for cond, val in conditions if val is not None)

        time_duration_conditions = [
            ("queued_duration >= :min_queued_duration", min_queued_duration),
            ("run_duration >= :min_run_duration", min_run_duration),
            (
                "total_upload_duration >= :min_total_upload_duration",
                min_total_upload_duration,
            ),
        ]

        where_time_duration_clause = " AND ".join(
            cond for cond, val in time_duration_conditions if val != 0
        )
        where_time_duration_clause = (
            f" AND {where_time_duration_clause}" if where_time_duration_clause else ""
        )

        # print("where_time_duration_clause", where_time_duration_clause)

        query = text(f"""
    WITH filtered_workflow_runs AS (
      SELECT
        id, status, created_at, queued_at, started_at, ended_at, gpu, workflow_id, machine_id, origin, workflow_version_id, user_id,
        CASE
          WHEN started_at IS NOT NULL AND created_at IS NOT NULL
          THEN EXTRACT(EPOCH FROM (started_at - created_at))
          ELSE NULL
        END AS queued_duration,
        CASE
          WHEN ended_at IS NOT NULL AND started_at IS NOT NULL
          THEN EXTRACT(EPOCH FROM (ended_at - started_at))
          ELSE NULL
        END AS run_duration
      FROM "comfyui_deploy"."workflow_runs"
      WHERE {where_clause}
    ),
    workflow_runs_with_upload_duration AS (
      SELECT
        fwr.*,
        CASE
          WHEN "comfyui_deploy"."workflow_run_outputs".data IS NOT NULL
          THEN ROUND(CAST(
            (SELECT COALESCE(SUM(
              CASE
                WHEN jsonb_typeof(value) = 'array' THEN
                  (SELECT COALESCE(SUM((elem->>'upload_duration')::float), 0)
                   FROM jsonb_array_elements(value) elem)
                WHEN jsonb_typeof(value) = 'object' THEN
                  COALESCE((value->>'upload_duration')::float, 0)
                ELSE 0
              END
            ), 0)
             FROM jsonb_each("comfyui_deploy"."workflow_run_outputs".data))
          AS NUMERIC), 5)
          ELSE NULL
        END AS total_upload_duration
      FROM filtered_workflow_runs fwr
      LEFT JOIN "comfyui_deploy"."workflow_run_outputs" ON fwr.id = "comfyui_deploy"."workflow_run_outputs".run_id
    )
    SELECT DISTINCT
      fwr.created_at,
      fwr.queued_at,
      fwr.started_at,
      fwr.ended_at,
      fwr.id AS run_id,
      fwr.gpu,
      fwr.origin,
      fwr.status AS run_status,
      fwr.workflow_id,
      "comfyui_deploy"."workflows".name AS workflow_name,
      "comfyui_deploy"."workflow_versions".version,
      fwr.machine_id,
      "comfyui_deploy"."machines".name AS machine_name,
      fwr.queued_duration,
      fwr.run_duration,
      fwr.total_upload_duration,
      fwr.user_id
    FROM workflow_runs_with_upload_duration AS fwr
    LEFT JOIN "comfyui_deploy"."workflows"
      ON fwr.workflow_id = "comfyui_deploy"."workflows".id
    LEFT JOIN "comfyui_deploy"."machines"
      ON fwr.machine_id = "comfyui_deploy"."machines".id
    LEFT JOIN "comfyui_deploy"."workflow_versions"
      ON fwr.workflow_version_id = "comfyui_deploy"."workflow_versions".id
    WHERE
      TRUE
      {where_time_duration_clause}
    ORDER BY fwr.created_at DESC
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
                "workflow_version_id": workflow_version_id,
                "min_queued_duration": min_queued_duration,
                "min_run_duration": min_run_duration,
                "min_total_upload_duration": min_total_upload_duration,
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
                "workflow_version": run.version,
                "machine": {"id": run.machine_id, "name": run.machine_name},
                "queued_duration": run.queued_duration,
                "run_duration": run.run_duration,
                "total_upload_duration": run.total_upload_duration,
                "user_id": run.user_id,
            }
            for run in runs
        ]

        return JSONResponse(
            status_code=200,
            content=json.loads(json.dumps(runs_data, cls=CustomJSONEncoder)),
        )
    except Exception as e:
        logger.error(f"Error getting runs: {e}")
        return JSONResponse(status_code=500, content={"Error. ": str(e)})


@router.get("/runs/timeline", response_model=List[WorkflowRunModel])
async def get_runs_time_line(
    request: Request,
    # filter time range
    start_time_unix: int = None,
    end_time_unix: int = None,
    # filter workflow
    workflow_id: Optional[List[str]] = Query(None),
    # filter status
    status: Optional[List[str]] = Query(None),
    # filter gpu
    gpu: Optional[List[str]] = Query(None),
    # filter machine
    machine_id: Optional[List[str]] = Query(None),
    # filter origin
    origin: Optional[List[str]] = Query(None),
    # filter workflow version
    workflow_version_id: Optional[List[str]] = Query(None),
    # filter queued duration
    min_queued_duration: Optional[str] = 0,
    # filter run duration
    min_run_duration: Optional[str] = 0,
    # filter total upload duration
    min_total_upload_duration: Optional[str] = 0,
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
        workflow_id = [clean_input(c) for c in workflow_id] if workflow_id else None
        status = [clean_input(s) for s in status] if status else None
        gpu = [clean_input(g) for g in gpu] if gpu else None
        machine_id = [clean_input(m) for m in machine_id] if machine_id else None
        origin = [clean_input(o) for o in origin] if origin else None
        workflow_version_id = (
            [clean_input(w) for w in workflow_version_id]
            if workflow_version_id
            else None
        )

        if min_queued_duration == "":
            min_queued_duration = 0
        if min_run_duration == "":
            min_run_duration = 0
        if min_total_upload_duration == "":
            min_total_upload_duration = 0

        conditions = [
            ("created_at >= :start_time", start_time),
            ("created_at <= :end_time", end_time),
            ("workflow_id = ANY(:workflow_id)", workflow_id),
            ("status = ANY(:status)", status),
            ("gpu = ANY(:gpu)", gpu),
            ("machine_id = ANY(:machine_id)", machine_id),
            ("origin = ANY(:origin)", origin),
            ("workflow_version_id = ANY(:workflow_version_id)", workflow_version_id),
        ]

        # Special handling for org_id and user_id
        if org_id:
            conditions.append(("org_id = :org_id", org_id))
        else:
            conditions.extend([
                ("org_id IS NULL", True),
                ("user_id = :user_id", user_id)
            ])

        where_clause = " AND ".join(cond for cond, val in conditions if val is not None)

        time_duration_conditions = [
            ("queued_duration >= :min_queued_duration", min_queued_duration),
            ("run_duration >= :min_run_duration", min_run_duration),
            (
                "total_upload_duration >= :min_total_upload_duration",
                min_total_upload_duration,
            ),
        ]

        where_time_duration_clause = " AND ".join(
            cond for cond, val in time_duration_conditions if val != 0
        )
        where_time_duration_clause = (
            f" AND {where_time_duration_clause}" if where_time_duration_clause else ""
        )

        # print("where_time_duration_clause", where_time_duration_clause)

        query = text(f"""
    WITH filtered_workflow_runs AS (
      SELECT
        id, status, created_at, queued_at, started_at, ended_at, gpu, workflow_id, machine_id, origin, workflow_version_id,
        CASE
          WHEN started_at IS NOT NULL AND created_at IS NOT NULL
          THEN EXTRACT(EPOCH FROM (started_at - created_at))
          ELSE NULL
        END AS queued_duration,
        CASE
          WHEN ended_at IS NOT NULL AND started_at IS NOT NULL
          THEN EXTRACT(EPOCH FROM (ended_at - started_at))
          ELSE NULL
        END AS run_duration
      FROM "comfyui_deploy"."workflow_runs"
      WHERE {where_clause}
    ),
    workflow_runs_with_upload_duration AS (
      SELECT
        fwr.*,
        CASE
          WHEN "comfyui_deploy"."workflow_run_outputs".data IS NOT NULL
          THEN ROUND(CAST(
            (SELECT COALESCE(SUM(
              CASE
                WHEN jsonb_typeof(value) = 'array' THEN
                  (SELECT COALESCE(SUM((elem->>'upload_duration')::float), 0)
                   FROM jsonb_array_elements(value) elem)
                WHEN jsonb_typeof(value) = 'object' THEN
                  COALESCE((value->>'upload_duration')::float, 0)
                ELSE 0
              END
            ), 0)
             FROM jsonb_each("comfyui_deploy"."workflow_run_outputs".data))
          AS NUMERIC), 5)
          ELSE NULL
        END AS total_upload_duration
      FROM filtered_workflow_runs fwr
      LEFT JOIN "comfyui_deploy"."workflow_run_outputs" ON fwr.id = "comfyui_deploy"."workflow_run_outputs".run_id
    )
    SELECT 
      fwr.created_at,
      fwr.id AS run_id,
      fwr.status AS run_status
    FROM 
      workflow_runs_with_upload_duration fwr
    WHERE
      TRUE
      {where_time_duration_clause}
    ORDER BY fwr.created_at DESC
    """)

        params = {
            key: val
            for key, val in {
                "start_time": start_time,
                "end_time": end_time,
                "org_id": org_id,
                "user_id": user_id,
                "workflow_id": workflow_id,
                "status": status,
                "gpu": gpu,
                "machine_id": machine_id,
                "origin": origin,
                "workflow_version_id": workflow_version_id,
                "min_queued_duration": min_queued_duration,
                "min_run_duration": min_run_duration,
                "min_total_upload_duration": min_total_upload_duration,
            }.items()
            if val is not None
        }

        result = await db.execute(query, params)
        runs = result.fetchall()

        runs_data = [
            {
                "created_at": run.created_at,
                "id": run.run_id,
                "status": run.run_status,
            }
            for run in runs
        ]

        return JSONResponse(
            status_code=200,
            content=json.loads(json.dumps(runs_data, cls=CustomJSONEncoder)),
        )
    except Exception as e:
        logger.error(f"Error getting runs: {e}")
        return JSONResponse(status_code=500, content={"Error. ": str(e)})
