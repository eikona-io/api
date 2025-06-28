from fastapi import APIRouter, HTTPException, Request, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy import func, and_, or_
from typing import List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime

from api.database import get_db
from api.models import OutputShare, WorkflowRun, User, WorkflowRunOutput
from api.routes.utils import select

router = APIRouter()

class OutputShareCreate(BaseModel):
    run_id: uuid.UUID
    output_id: uuid.UUID
    output_type: str = "other"
    visibility: str = "private"

class OutputShareResponse(BaseModel):
    id: uuid.UUID
    user_id: str
    org_id: Optional[str]
    run_id: uuid.UUID
    output_id: uuid.UUID
    output_data: dict
    output_type: str
    visibility: str
    created_at: datetime
    updated_at: datetime

def determine_output_type(output_data: dict) -> str:
    """Determine output type based on output data"""
    if not output_data:
        return "other"
    
    if any(key.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')) for key in output_data.keys() if isinstance(key, str)):
        return "image"
    
    if any(key.lower().endswith(('.mp4', '.avi', '.mov', '.webm')) for key in output_data.keys() if isinstance(key, str)):
        return "video"
    
    if any(key.lower().endswith(('.obj', '.fbx', '.gltf', '.glb', '.ply')) for key in output_data.keys() if isinstance(key, str)):
        return "3d"
    
    return "other"

@router.post("/share/output", response_model=OutputShareResponse)
async def create_output_share(
    share_data: OutputShareCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    run_query = select(WorkflowRun).where(
        and_(
            WorkflowRun.id == share_data.run_id,
            WorkflowRun.user_id == user.id
        )
    )
    run_result = await db.execute(run_query)
    run = run_result.scalar_one_or_none()
    
    if not run:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    
    output_query = select(WorkflowRunOutput).where(
        and_(
            WorkflowRunOutput.id == share_data.output_id,
            WorkflowRunOutput.run_id == share_data.run_id
        )
    )
    output_result = await db.execute(output_query)
    output = output_result.scalar_one_or_none()
    
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")
    
    if share_data.output_type == "other":
        detected_type = determine_output_type(output.data or {})
        share_data.output_type = detected_type
    
    output_share = OutputShare(
        user_id=user.id,
        org_id=getattr(user, 'org_id', None),
        run_id=share_data.run_id,
        output_id=share_data.output_id,
        output_data=output.data or {},
        output_type=share_data.output_type,
        visibility=share_data.visibility
    )
    
    db.add(output_share)
    await db.commit()
    await db.refresh(output_share)
    
    return OutputShareResponse(
        id=output_share.id,
        user_id=output_share.user_id,
        org_id=output_share.org_id,
        run_id=output_share.run_id,
        output_id=output_share.output_id,
        output_data=output_share.output_data,
        output_type=output_share.output_type,
        visibility=output_share.visibility,
        created_at=output_share.created_at,
        updated_at=output_share.updated_at
    )

@router.get("/share/output", response_model=List[OutputShareResponse])
async def list_output_shares(
    request: Request,
    output_type: Optional[str] = Query(None, description="Filter by output type"),
    visibility: Optional[str] = Query(None, description="Filter by visibility"),
    include_public: bool = Query(True, description="Include public shares for authenticated users"),
    db: AsyncSession = Depends(get_db)
):
    user = getattr(request.state, "user", None)
    
    query = select(OutputShare)
    conditions = []
    
    if user:
        org_condition = and_(
            OutputShare.user_id == user.id,
            or_(
                OutputShare.visibility == "private",
                OutputShare.visibility == "link"
            )
        )
        
        if include_public:
            conditions.append(or_(
                org_condition,
                OutputShare.visibility == "public"
            ))
        else:
            conditions.append(org_condition)
    else:
        conditions.append(OutputShare.visibility == "public")
    
    if output_type:
        conditions.append(OutputShare.output_type == output_type)
    
    if visibility:
        conditions.append(OutputShare.visibility == visibility)
    
    if conditions:
        query = query.where(and_(*conditions))
    
    query = query.order_by(OutputShare.created_at.desc())
    
    result = await db.execute(query)
    shares = result.scalars().all()
    
    return [
        OutputShareResponse(
            id=share.id,
            user_id=share.user_id,
            org_id=share.org_id,
            run_id=share.run_id,
            output_id=share.output_id,
            output_data=share.output_data,
            output_type=share.output_type,
            visibility=share.visibility,
            created_at=share.created_at,
            updated_at=share.updated_at
        )
        for share in shares
    ]

@router.get("/share/output/{share_id}")
async def get_shared_output(
    share_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    user = getattr(request.state, "user", None)
    
    query = select(OutputShare).options(
        joinedload(OutputShare.run),
        joinedload(OutputShare.user),
        joinedload(OutputShare.output)
    ).where(OutputShare.id == share_id)
    
    result = await db.execute(query)
    share = result.scalar_one_or_none()
    
    if not share:
        raise HTTPException(status_code=404, detail="Shared output not found")
    
    if share.visibility == "private":
        if not user or user.id != share.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif share.visibility == "link":
        pass
    elif share.visibility == "public":
        pass
    
    return {
        "share": OutputShareResponse(
            id=share.id,
            user_id=share.user_id,
            org_id=share.org_id,
            run_id=share.run_id,
            output_id=share.output_id,
            output_data=share.output_data,
            output_type=share.output_type,
            visibility=share.visibility,
            created_at=share.created_at,
            updated_at=share.updated_at
        ),
        "run": {
            "id": share.run.id,
            "status": share.run.status,
            "created_at": share.run.created_at
        } if share.run else None
    }

@router.delete("/share/output/{share_id}")
async def delete_output_share(
    share_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    query = select(OutputShare).where(
        and_(
            OutputShare.id == share_id,
            OutputShare.user_id == user.id
        )
    )
    result = await db.execute(query)
    share = result.scalar_one_or_none()
    
    if not share:
        raise HTTPException(status_code=404, detail="Output share not found")
    
    await db.delete(share)
    await db.commit()
    
    return {"message": "Output share deleted successfully"}
