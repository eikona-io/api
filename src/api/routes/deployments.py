import json
import logging
import uuid
from api.routes.run import get_comfy_deploy_runner, redeploy_comfy_deploy_runner_if_exists
from api.utils.inputs import get_inputs_from_workflow_api
from api.utils.outputs import get_outputs_from_workflow
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Any, Dict, List, Optional, Union
from .types import DeploymentModel, DeploymentEnvironment
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import select
from api.models import Deployment, Machine, MachineVersion, Workflow
from api.database import get_db
from fastapi.responses import JSONResponse
from sqlalchemy.orm import joinedload
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Deployments"])

class DeploymentCreate(BaseModel):
    workflow_version_id: str
    workflow_id: str
    machine_id: Optional[str] = None
    machine_version_id: Optional[str] = None
    environment: str

@router.post(
    "/deployment",
    response_model=DeploymentModel,
    openapi_extra={
        "x-speakeasy-name-override": "create",
    },
)
async def create_deployment(
    request: Request,
    deployment_data: DeploymentCreate,
    db: AsyncSession = Depends(get_db),
):
    user_id = request.state.current_user["user_id"]
    org_id = (
        request.state.current_user["org_id"]
        if "org_id" in request.state.current_user
        else None
    )
    
    if deployment_data.machine_id is None and deployment_data.machine_version_id is None:
        raise HTTPException(status_code=400, detail="Machine ID or Machine Version ID is required")
    
    try:
        # Check for existing deployment with same environment
        existing_deployment_query = select(Deployment).where(
            Deployment.workflow_id == deployment_data.workflow_id,
            Deployment.environment == deployment_data.environment
        ).apply_org_check(request)
            
        result = await db.execute(existing_deployment_query)
        existing_deployment = result.scalar_one_or_none()
        
        machine_version = None
        machine_id = deployment_data.machine_id
    
        if deployment_data.machine_version_id is not None:
            machine_version_query = select(MachineVersion).where(MachineVersion.id == deployment_data.machine_version_id)
            result = await db.execute(machine_version_query)
            machine_version = result.scalar_one_or_none()
            machine_id = machine_version.machine_id
            
        # Get current machine and machine version
        machine_query = select(Machine).where(Machine.id == machine_id)
        result = await db.execute(machine_query)
        machine = result.scalar_one_or_none()
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")

        if machine.machine_version_id and machine_version is None:
            machine_version_query = select(MachineVersion).where(MachineVersion.id == machine.machine_version_id)
            result = await db.execute(machine_version_query)
            machine_version = result.scalar_one_or_none()

        if existing_deployment:
            # Update existing deployment
            existing_deployment.workflow_version_id = deployment_data.workflow_version_id
            existing_deployment.machine_id = machine_id
            deployment = existing_deployment
            
            if machine_version is not None and machine_version.modal_image_id is not None:
                existing_deployment.machine_version_id = machine_version.id
                existing_deployment.modal_image_id = machine_version.modal_image_id
                existing_deployment.gpu = machine_version.gpu
                existing_deployment.run_timeout = machine_version.run_timeout
                existing_deployment.idle_timeout = machine_version.idle_timeout
                
                if (existing_deployment.modal_image_id != machine_version.modal_image_id):
                    # We should trigger a redeploy
                    await redeploy_comfy_deploy_runner_if_exists(machine_id, machine_version.gpu, existing_deployment)
                
                # Not supporting this for now
                # existing_deployment.keep_warm = machine_version.keep_warm
            
        else:
            # Create new deployment object
            deployment = Deployment(
                id=uuid.uuid4(),
                user_id=user_id,
                org_id=org_id,
                workflow_version_id=deployment_data.workflow_version_id,
                workflow_id=deployment_data.workflow_id,
                machine_id=machine_id,
                environment=deployment_data.environment,
            )
            if machine_version is not None and machine_version.modal_image_id is not None:
                deployment.machine_version_id = machine_version.id
                deployment.modal_image_id = machine_version.modal_image_id
                deployment.gpu = machine_version.gpu
                deployment.run_timeout = machine_version.run_timeout
                deployment.idle_timeout = machine_version.idle_timeout
                # Not supporting this for now
                # deployment.keep_warm = machine_version.keep_warm
            db.add(deployment)

        await db.commit()
        await db.refresh(deployment)

        # Convert to dict
        deployment_dict = deployment.to_dict()
        return deployment_dict

    except Exception as e:
        logger.error(f"Error creating deployment: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")



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
    try:
        query = select(Deployment).options(
            joinedload(Deployment.workflow).load_only(Workflow.name),
            joinedload(Deployment.version),
        ).join(Workflow).where(Workflow.deleted == False)

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
    except Exception as e:
        logger.error(f"Error getting deployments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
