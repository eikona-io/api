import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional
from .types import WorkflowListResponse, WorkflowModel
from api.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import select, post_process_outputs
from api.models import Workflow
from .utils import get_user_settings
from sqlalchemy import func

logger = logging.getLogger(__name__)

router = APIRouter(tags=["workflows"])


@router.get("/workflows", response_model=List[WorkflowModel])
async def get_workflows(
    request: Request,
    page: str = "1",
    search: Optional[str] = None,
    limit: int = 8,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    # org_id = request.state.current_user.get("org_id")
    # user_id = request.state.current_user.get("user_id")

    # logger.info(f"User {user_id} org {org_id} requesting workflows")

    # get the count of workflows
    # count_query = select(func.count()).select_from(Workflow)

    query = (
        select(Workflow)
        .order_by(Workflow.pinned.desc())
        .order_by(Workflow.updated_at.desc())
        .apply_org_check(request)
        .paginate(limit, offset)
    )

    if search:
        query = query.where(func.lower(Workflow.name).ilike(f"%{search.lower()}%"))

    # query_length = await db.execute(count_query)
    result = await db.execute(query)
    workflows = result.scalars().all()

    if not workflows:
        # raise HTTPException(status_code=404, detail="Workflows not found")
        return []
    workflows_data = [workflow.to_dict() for workflow in workflows]

    return workflows_data
