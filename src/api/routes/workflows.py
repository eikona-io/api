import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional
from .types import WorkflowListResponse, WorkflowModel
from api.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import select, post_process_outputs
from api.models import Deployment, User, Workflow, WorkflowRun, WorkflowVersion
from .utils import get_user_settings
from sqlalchemy import func, select as sa_select, distinct, and_, or_
from sqlalchemy.orm import joinedload, load_only, contains_eager
from fastapi.responses import JSONResponse
from pprint import pprint
from sqlalchemy import desc

logger = logging.getLogger(__name__)

router = APIRouter(tags=["workflows"])


@router.get("/workflows", response_model=List[WorkflowModel])
async def get_workflows(
    request: Request,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    # Subquery to get the latest version for each workflow
    latest_versions = (
        sa_select(
            WorkflowVersion.workflow_id,
            func.max(WorkflowVersion.version).label("latest_version"),
        )
        .group_by(WorkflowVersion.workflow_id)
        .subquery()
    )

    # Subquery to get the latest public deployment for each workflow
    latest_deployments = (
        sa_select(
            Deployment.workflow_id,
            func.max(Deployment.id).label("latest_deployment_id"),
        )
        .where(Deployment.environment == "public-share")
        .group_by(Deployment.workflow_id)
        .subquery()
    )

    # Subquery to get the latest run for each workflow
    latest_runs = (
        sa_select(
            WorkflowRun.workflow_id,
            func.max(WorkflowRun.created_at).label("latest_run_date"),
        )
        .group_by(WorkflowRun.workflow_id)
        .subquery()
    )

    # Subquery to get the latest run with output for each workflow
    latest_runs_with_output = (
        sa_select(
            WorkflowRun.workflow_id,
            WorkflowRun.id.label("latest_run_id"),
            WorkflowRun.outputs.label("latest_output"),
        )
        .distinct(WorkflowRun.workflow_id)
        .order_by(WorkflowRun.workflow_id, desc(WorkflowRun.created_at))
        .subquery()
    )

    query = (
        select(Workflow)
        .options(
            joinedload(Workflow.user).load_only(User.name),
            # joinedload(Workflow.versions).load_only(WorkflowVersion.id, WorkflowVersion.version),
            contains_eager(Workflow.versions).load_only(
                WorkflowVersion.id, WorkflowVersion.version
            ),
            joinedload(Workflow.deployments),
            # contains_eager(Workflow.runs).load_only(
            #     WorkflowRun.id, WorkflowRun.created_at, WorkflowRun.status
            # ),
            contains_eager(Workflow.runs).load_only(
                WorkflowRun.id,
                WorkflowRun.created_at,
                WorkflowRun.status,
                # WorkflowRun.outputs,
            ).selectinload(WorkflowRun.outputs)  # Add this line to load the outputs
        )
        .outerjoin(latest_versions, Workflow.id == latest_versions.c.workflow_id)
        .outerjoin(
            WorkflowVersion,
            and_(
                WorkflowVersion.workflow_id == Workflow.id,
                WorkflowVersion.version == latest_versions.c.latest_version,
            ),
        )
        # .outerjoin(latest_deployments, Workflow.id == latest_deployments.c.workflow_id)
        # .outerjoin(Deployment,
        #           and_(Deployment.workflow_id == Workflow.id,
        #                Deployment.id == latest_deployments.c.latest_deployment_id))
        # .outerjoin(latest_runs, Workflow.id == latest_runs.c.workflow_id)
        # .outerjoin(
        #     WorkflowRun,
        #     and_(
        #         WorkflowRun.workflow_id == Workflow.id,
        #         WorkflowRun.created_at == latest_runs.c.latest_run_date,
        #     ),
        # )
        .outerjoin(
            latest_runs_with_output,
            Workflow.id == latest_runs_with_output.c.workflow_id,
        )
        .outerjoin(
            WorkflowRun,
            and_(
                WorkflowRun.workflow_id == Workflow.id,
                WorkflowRun.id == latest_runs_with_output.c.latest_run_id,
            ),
        )
        .order_by(Workflow.pinned.desc(), Workflow.updated_at.desc())
        .apply_org_check(request)
        .limit(limit)
        .offset(offset)
        .distinct()
    )

    if search:
        query = query.where(func.lower(Workflow.name).ilike(f"%{search.lower()}%"))

    # Execute the query
    result = await db.execute(query)

    # Fetch the results
    workflows = result.unique().scalars().all()

    # pprint(workflows)

    if not workflows:
        # raise HTTPException(status_code=404, detail="Workflows not found")
        return []
    workflows_data = [workflow.to_dict() for workflow in workflows]

    # workflows_data = [
    #     {
    #         **workflow.to_dict(),
    #         "latest_output": workflow.runs[0].output if workflow.runs else None,
    #     }
    #     for workflow in workflows
    # ]

    return JSONResponse(status_code=200, content=workflows_data)
