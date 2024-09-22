import logging
from fastapi import APIRouter, Depends, Request
from typing import List
from .types import DeploymentModel, DeploymentEnvironment
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import select
from api.models import Deployment, Workflow
from api.database import get_db
from fastapi.responses import JSONResponse
from sqlalchemy.orm import joinedload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Deployments"])


@router.get("/deployments", response_model=List[DeploymentModel])
async def get_deployments(
    request: Request,
    environment: DeploymentEnvironment,
    db: AsyncSession = Depends(get_db),
):
    print("environment", environment)

    query = (
        select(Deployment)
        .where(Deployment.environment == environment)
        .options(joinedload(Deployment.workflow).load_only(Workflow.name))
        .apply_org_check(request)
    )

    result = await db.execute(query)
    deployments = result.scalars().all()


    deployments_data = [deployment.to_dict() for deployment in deployments]

    return JSONResponse(content=deployments_data)
