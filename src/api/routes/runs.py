import logging
from typing import List, Optional
from .types import WorkflowRunModel
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db, AsyncSessionLocal
from .utils import select
from api.models import WorkflowRun, Workflow, WorkflowVersion, Machine
from fastapi.responses import JSONResponse
from datetime import datetime
from sqlalchemy import func, case
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Runs"])


@router.get("/runs", response_model=List[WorkflowRunModel])
async def get_runs(
    request: Request,
    limit: int = 60,
    offset: int = 0,
    gpu: Optional[str] = None,
    status: Optional[str] = None,
    origin: Optional[str] = None,
    workflow_id: Optional[str] = None,
    duration: Optional[str] = None,
    created_at: Optional[str] = None,
    machine_id: Optional[str] = None,
):
    # Process filter lists upfront
    gpu_list = [g.strip() for g in gpu.split(",")] if gpu else []
    status_list = [s.strip().lower() for s in status.split(",")] if status else []
    origin_list = [o.strip() for o in origin.split(",")] if origin else []

    # Process datetime variables upfront
    start_datetime = None
    end_datetime = None
    if created_at:
        try:
            # Validate format
            if "-" not in created_at:
                raise ValueError("Time range must be in format 'start-end'")

            start_time, end_time = created_at.split("-")

            # Convert to timestamps
            start_datetime = datetime.fromtimestamp(int(start_time) / 1000)
            end_datetime = datetime.fromtimestamp(int(end_time) / 1000)

            # Validate time range
            if start_datetime > end_datetime:
                raise ValueError("Start time cannot be later than end time")

        except (ValueError, TypeError) as e:
            # Return error response instead of just logging
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid time range format: {str(e)}"},
            )

    # Get total count first (unfiltered)
    total_count_query = (
        select(func.count(WorkflowRun.id))
        .select_from(WorkflowRun)
        .filter(WorkflowRun.workflow_id.isnot(None))
        .apply_org_check(request)
    )

    # Create base query with joins
    base_query = (
        select(
            WorkflowRun.id,
            WorkflowRun.status,
            WorkflowRun.created_at,
            WorkflowRun.started_at,
            WorkflowRun.ended_at,
            WorkflowRun.workflow_id,
            WorkflowRun.workflow_version_id,
            WorkflowRun.machine_id,
            WorkflowRun.gpu,
            WorkflowRun.origin,
            WorkflowRun.user_id,
            case(
                (
                    WorkflowRun.ended_at.isnot(None)
                    & WorkflowRun.started_at.isnot(None),
                    func.extract(
                        "epoch", WorkflowRun.ended_at - WorkflowRun.started_at
                    ),
                ),
                else_=None,
            ).label("duration"),
        )
        .select_from(WorkflowRun)
        .filter(WorkflowRun.workflow_id.isnot(None))
        .apply_org_check(request)
    )

    # Handle filters
    if gpu_list:
        base_query = base_query.filter(WorkflowRun.gpu.in_(gpu_list))

    if status_list:
        base_query = base_query.filter(WorkflowRun.status.in_(status_list))

    if origin_list:
        base_query = base_query.filter(WorkflowRun.origin.in_(origin_list))

    if workflow_id:
        base_query = base_query.filter(WorkflowRun.workflow_id == workflow_id)

    if machine_id:
        base_query = base_query.filter(WorkflowRun.machine_id == machine_id)

    if start_datetime and end_datetime:
        base_query = base_query.filter(
            WorkflowRun.created_at.between(start_datetime, end_datetime)
        )

    # Add duration filter
    if duration:
        try:
            start_duration, end_duration = map(float, duration.split("-"))
            base_query = base_query.filter(
                WorkflowRun.ended_at.isnot(None),
                WorkflowRun.started_at.isnot(None),
                func.extract(
                    "epoch", WorkflowRun.ended_at - WorkflowRun.started_at
                ).between(start_duration, end_duration),
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid duration range format: {e}")

    # Check if any filters are applied
    has_filters = any(
        [
            gpu_list,
            status_list,
            origin_list,
            workflow_id,
            machine_id,
            start_datetime and end_datetime,  # Only count if both dates are present
            duration,
        ]
    )

    async def fetch_total_count():
        async with AsyncSessionLocal() as count_db:
            return await count_db.scalar(total_count_query)

    async def fetch_filter_count():
        if not has_filters:
            return None  # Will use total_count instead if no filters
        async with AsyncSessionLocal() as count_db:
            filter_count_query = select(func.count()).select_from(base_query.subquery())
            return await count_db.scalar(filter_count_query)

    async def fetch_runs_data():
        async with AsyncSessionLocal() as runs_db:
            # Get paginated data
            query = base_query.order_by(WorkflowRun.created_at.desc()).paginate(
                limit, offset
            )
            result = await runs_db.execute(query)
            runs = result.unique().all()

            if not runs:
                return []

            runs_data = []
            for (
                id,
                status,
                created_at,
                started_at,
                ended_at,
                workflow_id,
                workflow_version_id,
                machine_id,
                gpu,
                origin,
                user_id,
                duration,
            ) in runs:
                run_dict = {
                    "id": str(id),
                    "status": status,
                    "created_at": created_at.isoformat() if created_at else None,
                    "started_at": started_at.isoformat() if started_at else None,
                    "ended_at": ended_at.isoformat() if ended_at else None,
                    "workflow_id": str(workflow_id),
                    "workflow_version_id": str(workflow_version_id)
                    if workflow_version_id
                    else None,
                    "machine_id": str(machine_id) if machine_id else None,
                    "gpu": gpu,
                    "origin": origin,
                    "user_id": str(user_id) if user_id else None,
                    "duration": str(duration) if duration else None,
                }
                runs_data.append(run_dict)
            return runs_data

    async def fetch_chart_data():
        async with AsyncSessionLocal() as chart_db:
            chart_query = (
                select(
                    func.date_trunc("hour", WorkflowRun.created_at).label("hour"),
                    WorkflowRun.status,
                    func.count().label("count"),
                )
                .select_from(WorkflowRun)
                .apply_org_check(request)
            )

            # Apply all filters to chart query
            if gpu_list:
                chart_query = chart_query.filter(WorkflowRun.gpu.in_(gpu_list))
            if status_list:
                chart_query = chart_query.filter(WorkflowRun.status.in_(status_list))
            if origin_list:
                chart_query = chart_query.filter(WorkflowRun.origin.in_(origin_list))
            if workflow_id:
                chart_query = chart_query.filter(WorkflowRun.workflow_id == workflow_id)
            if machine_id:
                chart_query = chart_query.filter(WorkflowRun.machine_id == machine_id)
            if start_datetime and end_datetime:
                chart_query = chart_query.filter(
                    WorkflowRun.created_at.between(start_datetime, end_datetime)
                )
            if duration:
                start_duration, end_duration = map(float, duration.split("-"))
                chart_query = chart_query.filter(
                    WorkflowRun.ended_at.isnot(None),
                    WorkflowRun.started_at.isnot(None),
                    func.extract(
                        "epoch", WorkflowRun.ended_at - WorkflowRun.started_at
                    ).between(start_duration, end_duration),
                )

            chart_query = chart_query.group_by("hour", WorkflowRun.status).order_by(
                "hour"
            )
            chart_result = await chart_db.execute(chart_query)
            chart_rows = chart_result.all()

            # Process chart data using defaultdict
            chart_data = defaultdict(
                lambda: {"success": 0, "failed": 0, "others": 0, "timestamp": None}
            )

            for hour, status, count in chart_rows:
                timestamp = int(hour.timestamp() * 1000)
                chart_data[timestamp]["timestamp"] = timestamp

                if status.lower() == "success":
                    chart_data[timestamp]["success"] += count
                elif status.lower() == "failed":
                    chart_data[timestamp]["failed"] += count
                else:
                    chart_data[timestamp]["others"] += count

            return list(chart_data.values())

    # Run all four queries in parallel
    total_count, filter_count, runs_data, chart_data = await asyncio.gather(
        fetch_total_count(), fetch_filter_count(), fetch_runs_data(), fetch_chart_data()
    )

    return JSONResponse(
        content={
            "data": runs_data,
            "meta": {
                "totalRowCount": total_count,
                "filterRowCount": filter_count
                or total_count,  # Use total_count as fallback
                "chartData": chart_data,
            },
        }
    )
