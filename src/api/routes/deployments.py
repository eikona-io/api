import json
import logging
from api.utils.inputs import get_inputs_from_workflow_api
from api.utils.outputs import get_outputs_from_workflow
from fastapi import APIRouter, Depends, Request
from typing import Any, Dict, List, Optional, Union
from .types import DeploymentModel, DeploymentEnvironment
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import select
from api.models import Deployment, Workflow
from api.database import get_db
from fastapi.responses import JSONResponse
from sqlalchemy.orm import joinedload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Deployments"])


@router.get(
    "/deployments",
    response_model=List[DeploymentModel],
    openapi_extra={
        "x-speakeasy-name-override": "list",
    },
)
async def get_deployments(
    request: Request,
    environment: Optional[DeploymentEnvironment] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Deployment).options(
        joinedload(Deployment.workflow).load_only(Workflow.name),
        joinedload(Deployment.version),
    )

    if environment is not None:
        query = query.where(Deployment.environment == environment)

    query = query.apply_org_check(request)

    result = await db.execute(query)
    deployments = result.scalars().all()

    deployments_data = []
    for deployment in deployments:
        deployment_dict = deployment.to_dict()
        workflow_api = deployment.version.workflow_api if deployment.version else None
        inputs = get_inputs_from_workflow_api(workflow_api)
        
        workflow = deployment.version.workflow if deployment.version else None
        outputs = get_outputs_from_workflow(workflow)
        
        if inputs:
            deployment_dict["input_types"] = inputs

        if outputs:
            deployment_dict["output_types"] = outputs

        deployments_data.append(deployment_dict)

    return deployments_data
