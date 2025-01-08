import logging
from typing import List, Optional
from .types import WorkflowRunModel
from fastapi import APIRouter, Depends, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from .utils import select
from api.models import WorkflowRun, Workflow, WorkflowVersion, Machine
from fastapi.responses import JSONResponse
from .utils import ensure_run_timeout
from datetime import datetime, timedelta
from sqlalchemy import func, case
from sqlalchemy.types import String

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Runs"])


@router.get("/runs", response_model=List[WorkflowRunModel])
async def get_runs(
    request: Request,
    limit: int = 30,
    offset: int = 0,
    gpu: Optional[str] = None,
    status: Optional[str] = None,
    origin: Optional[str] = None,
    workflow_id: Optional[str] = None,
    duration: Optional[str] = None,
    created_at: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    # Get total count first (unfiltered)
    total_count_query = select(func.count()).select_from(
        select(WorkflowRun)
        .join(Workflow, WorkflowRun.workflow_id == Workflow.id)
        .apply_org_check(request)
        .subquery()
    )
    total_count = await db.scalar(total_count_query)

    # Create base query with joins
    base_query = (
        select(
            WorkflowRun,
            Workflow.name.label("workflow_name"),
            WorkflowVersion.version.label("workflow_version"),
            Machine.name.label("machine_name"),
            case(
                (
                    (
                        WorkflowRun.ended_at.isnot(None)
                        & WorkflowRun.started_at.isnot(None)
                    ),
                    func.cast(
                        func.extract(
                            "epoch", WorkflowRun.ended_at - WorkflowRun.started_at
                        ),
                        String,
                    ),
                ),
                else_=None,
            ).label("duration"),
        )
        .join(
            Workflow, WorkflowRun.workflow_id == Workflow.id
        )  # Caution: it will filter the workflow without workflow_id. make it outerjoin if you want to keep empty workflow_id
        .outerjoin(
            WorkflowVersion, WorkflowRun.workflow_version_id == WorkflowVersion.id
        )
        .outerjoin(Machine, WorkflowRun.machine_id == Machine.id)
        .apply_org_check(request)
    )

    # Handle filters
    if gpu:
        gpu_list = [g.strip() for g in gpu.split(",")]
        base_query = base_query.filter(WorkflowRun.gpu.in_(gpu_list))

    if status:
        status_list = [s.strip().lower() for s in status.split(",")]
        base_query = base_query.filter(WorkflowRun.status.in_(status_list))

    if origin:
        origin_list = [o.strip() for o in origin.split(",")]
        base_query = base_query.filter(WorkflowRun.origin.in_(origin_list))

    if workflow_id:
        base_query = base_query.filter(WorkflowRun.workflow_id == workflow_id)

    if created_at:
        try:
            start_time, end_time = created_at.split("-")
            start_datetime = datetime.fromtimestamp(int(start_time) / 1000)
            end_datetime = datetime.fromtimestamp(int(end_time) / 1000)
            base_query = base_query.filter(
                WorkflowRun.created_at.between(start_datetime, end_datetime)
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid time range format: {e}")

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

    # Get filtered count
    filter_count_query = select(func.count()).select_from(base_query.subquery())
    filter_count = await db.scalar(filter_count_query)

    # Get paginated data
    query = base_query.order_by(WorkflowRun.created_at.desc()).paginate(limit, offset)
    result = await db.execute(query)
    runs = result.unique().all()

    if not runs:
        return JSONResponse(
            content={
                "data": [],
                "meta": {"totalRowCount": total_count, "filterRowCount": filter_count},
            }
        )

    runs_data = []
    for run, workflow_name, workflow_version, machine_name, duration in runs:
        ensure_run_timeout(run)
        run_dict = run.to_dict()
        run_dict.update(
            {
                "workflow": {"id": str(run.workflow_id), "name": workflow_name},
                "workflow_version": workflow_version,
                "machine": {
                    "id": str(run.machine_id) if run.machine_id else None,
                    "name": machine_name,
                },
                "duration": duration,
            }
        )
        runs_data.append(run_dict)

    # Get chart data
    chart_query = (
        select(
            func.date_trunc("hour", WorkflowRun.created_at).label("hour"),
            WorkflowRun.status,
            func.count().label("count"),
        )
        .select_from(WorkflowRun)
        .apply_org_check(request)
    )

    # Apply the same filters as the main query
    if gpu:
        chart_query = chart_query.filter(WorkflowRun.gpu.in_(gpu_list))
    if origin:
        chart_query = chart_query.filter(WorkflowRun.origin.in_(origin_list))
    if created_at:
        try:
            chart_query = chart_query.filter(
                WorkflowRun.created_at.between(start_datetime, end_datetime)
            )
        except (ValueError, TypeError):
            pass

    chart_query = chart_query.group_by("hour", WorkflowRun.status).order_by("hour")

    chart_result = await db.execute(chart_query)
    chart_rows = chart_result.all()

    # Process chart data
    chart_data = {}
    for hour, status, count in chart_rows:
        timestamp = int(hour.timestamp() * 1000)
        if timestamp not in chart_data:
            chart_data[timestamp] = {
                "success": 0,
                "failed": 0,
                "others": 0,
                "timestamp": timestamp,
            }

        # Categorize the status
        if status.lower() == "success":
            chart_data[timestamp]["success"] += count
        elif status.lower() == "failed":
            chart_data[timestamp]["failed"] += count
        else:
            chart_data[timestamp]["others"] += count

    return JSONResponse(
        content={
            "data": runs_data,
            "meta": {
                "totalRowCount": total_count,
                "filterRowCount": filter_count,
                "chartData": list(chart_data.values()),
            },
        }
    )
