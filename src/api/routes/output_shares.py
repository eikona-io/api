from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy import func
from typing import List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime
import secrets
import string

from api.database import get_db
from api.models import OutputShare, WorkflowRun, User, WorkflowRunOutput
from api.routes.utils import select

router = APIRouter()


class OutputShareCreate(BaseModel):
    run_id: uuid.UUID
    output_id: uuid.UUID
    visibility: str = "link-only"


class OutputShareResponse(BaseModel):
    id: uuid.UUID
    user_id: str
    org_id: Optional[str]
    run_id: uuid.UUID
    output_id: uuid.UUID
    output_data: dict
    share_slug: str
    visibility: str
    created_at: datetime
    updated_at: datetime


def generate_share_slug() -> str:
    return ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))


@router.post("/output-shares", response_model=OutputShareResponse)
async def create_output_share(
    request: Request,
    body: OutputShareCreate,
    db: AsyncSession = Depends(get_db),
):
    user_id = request.state.current_user.get("user_id")
    org_id = request.state.current_user.get("org_id")
    
    run_query = select(WorkflowRun).where(WorkflowRun.id == body.run_id).apply_org_check(request)
    result = await db.execute(run_query)
    run = result.scalar_one_or_none()
    
    if not run:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    
    from sqlalchemy import select as sql_select
    output_query = sql_select(WorkflowRunOutput).where(
        WorkflowRunOutput.id == body.output_id,
        WorkflowRunOutput.run_id == body.run_id
    )
    output_result = await db.execute(output_query)
    output = output_result.scalar_one_or_none()
    
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")
    
    share_slug = generate_share_slug()
    while True:
        from sqlalchemy import select as sql_select
        existing = await db.execute(sql_select(OutputShare).where(OutputShare.share_slug == share_slug))
        if not existing.scalar_one_or_none():
            break
        share_slug = generate_share_slug()
    
    output_share = OutputShare(
        user_id=user_id,
        org_id=org_id,
        run_id=body.run_id,
        output_id=body.output_id,
        output_data=output.data,
        share_slug=share_slug,
        visibility=body.visibility,
    )
    
    db.add(output_share)
    await db.commit()
    await db.refresh(output_share)
    
    return output_share.to_dict()


@router.get("/output-shares", response_model=List[OutputShareResponse])
async def list_output_shares(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    query = select(OutputShare).apply_org_check(request).order_by(OutputShare.created_at.desc())
    result = await db.execute(query)
    shares = result.scalars().all()
    
    return [share.to_dict() for share in shares]


@router.get("/output-shares/{slug}")
async def get_shared_outputs(
    slug: str,
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select as sql_select
    query = (
        sql_select(OutputShare)
        .options(joinedload(OutputShare.run))
        .where(OutputShare.share_slug == slug)
    )
    
    result = await db.execute(query)
    share = result.scalar_one_or_none()
    
    if not share:
        raise HTTPException(status_code=404, detail="Shared outputs not found")
    
    if share.visibility == "link-only":
        pass
    elif share.visibility == "public":
        pass
    elif share.visibility == "public-in-org":
        pass
    
    return {
        "share": share.to_dict(),
        "run": share.run.to_dict() if share.run else None,
    }


@router.delete("/output-shares/{share_id}")
async def delete_output_share(
    request: Request,
    share_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    query = select(OutputShare).where(OutputShare.id == share_id).apply_org_check(request)
    result = await db.execute(query)
    share = result.scalar_one_or_none()
    
    if not share:
        raise HTTPException(status_code=404, detail="Output share not found")
    
    await db.delete(share)
    await db.commit()
    
    return {"message": "Output share deleted successfully"}
