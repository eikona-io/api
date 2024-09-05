import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional
from .types import WorkflowListResponse
from api.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import select, post_process_outputs
from api.models import Workflow
from .utils import get_user_settings
from sqlalchemy import func

logger = logging.getLogger(__name__)

router = APIRouter(tags=["workflows"])


@router.get("/workflows", response_model=WorkflowListResponse)
async def get_workflows(
    request: Request,
    page: str = "1",
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):    
    org_id = request.state.current_user.get("org_id")
    user_id = request.state.current_user.get("user_id")
    
    logger.info(f"User {user_id} org {org_id} requesting workflows")
    
    # get the count of workflows
    count_query = select(func.count()).select_from(Workflow)
    
    query = (
        select(Workflow)
        .order_by(Workflow.pinned.desc())
        .order_by(Workflow.updated_at.desc())
        .limit(8)
        .offset((int(page) - 1) * 8)
    )

    if org_id:
        query = query.where(Workflow.org_id == org_id)
        count_query = count_query.where(Workflow.org_id == org_id)
    elif user_id:
        query = query.where(Workflow.user_id == user_id, Workflow.org_id.is_(None))
        count_query = count_query.where(Workflow.user_id == user_id, Workflow.org_id.is_(None))
    else:
        raise HTTPException(status_code=404, detail="No organization or user found")

    if search:
        query = query.where(Workflow.name.contains(search))
        count_query = count_query.where(Workflow.name.contains(search))

    query_length = await db.execute(count_query)
    result = await db.execute(query)
    workflows = result.scalars().all()

    if not workflows:
        raise HTTPException(status_code=404, detail="Workflows not found")

    return {"workflows": workflows, "query_length": query_length.scalar()}
