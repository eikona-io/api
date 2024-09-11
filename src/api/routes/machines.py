from http.client import HTTPException
from uuid import UUID
from .types import (
    MachineModel,
    WorkflowVersionModel,
)
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from .utils import select
from sqlalchemy import func
from fastapi.responses import JSONResponse

from api.models import Machine

# from sqlalchemy import select
from api.database import get_db
import logging
from typing import List, Optional
# from fastapi_pagination import Page, add_pagination, paginate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Machine"])


@router.get("/machines", response_model=List[MachineModel])
async def get_machines(
    request: Request,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    is_deleted: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
):
    machines_query = (
        select(Machine)
        .order_by(Machine.created_at.desc())
        .where(Machine.deleted == is_deleted)
        .apply_org_check(request)
        .paginate(limit, offset)
    )

    if search:
        machines_query = machines_query.where(
            func.lower(Machine.name).contains(search.lower())
        )

    result = await db.execute(machines_query)
    machines = result.unique().scalars().all()

    if not machines:
        # raise HTTPException(status_code=404, detail="Runs not found")
        return []

    machines_data = [machine.to_dict() for machine in machines]

    return JSONResponse(content=machines_data)


@router.get("/machine/{machine_id}", response_model=MachineModel)
async def get_machine(
    request: Request,
    machine_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    machine = await db.execute(select(Machine).where(Machine.id == machine_id))
    machine = machine.scalars().first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    return JSONResponse(content=machine.to_dict())
