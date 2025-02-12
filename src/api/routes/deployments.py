import logging
import uuid
from api.routes.run import redeploy_comfy_deploy_runner_if_exists
from api.utils.inputs import get_inputs_from_workflow_api
from api.utils.outputs import get_outputs_from_workflow
from api.routes.platform import get_clerk_data_with_slug
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional
from .types import DeploymentModel, DeploymentEnvironment, DeploymentShareModel
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import select
from api.models import Deployment, Machine, MachineVersion, User, Workflow
from api.database import get_db
from sqlalchemy.orm import joinedload
from pydantic import BaseModel
from enum import Enum
from api.modal.builder import GPUType, KeepWarmBody, set_machine_always_on

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Deployments"])

class GPUType(str, Enum):
    CPU = "CPU"
    T4 = "T4"
    L4 = "L4"
    A10G = "A10G"
    L40S = "L40S"
    A100 = "A100"
    A100_80GB = "A100-80GB"
    H100 = "H100"

class DeploymentCreate(BaseModel):
    workflow_version_id: str
    workflow_id: str
    machine_id: Optional[str] = None
    machine_version_id: Optional[str] = None
    environment: str
    description: Optional[str] = None
    share_slug: Optional[str] = None

class DeploymentUpdate(BaseModel):
    workflow_version_id: Optional[str] = None
    machine_id: Optional[str] = None
    machine_version_id: Optional[str] = None
    concurrency_limit: Optional[int] = None
    gpu: Optional[GPUType] = None
    run_timeout: Optional[int] = None
    idle_timeout: Optional[int] = None
    keep_warm: Optional[int] = None

async def update_deployment_with_machine(
    deployment: Deployment,
    machine_id: str,
    machine_version: Optional[MachineVersion],
    db: AsyncSession,
    update_data: Optional[DeploymentUpdate] = None,
) -> Deployment:
    """Update deployment with machine and machine version information."""
    # Store original values to check for changes
    original_modal_image_id = deployment.modal_image_id
    original_run_timeout = deployment.run_timeout
    original_idle_timeout = deployment.idle_timeout
    original_concurrency_limit = deployment.concurrency_limit
    original_keep_warm = deployment.keep_warm
    
    deployment.machine_id = machine_id
            
    if machine_version is not None and machine_version.modal_image_id is not None:
        deployment.machine_version_id = machine_version.id
        deployment.modal_image_id = machine_version.modal_image_id
        
        # Only update these fields from machine_version if not provided in update_data
        if update_data is None or update_data.gpu is None:
            deployment.gpu = machine_version.gpu
        if update_data is None or update_data.run_timeout is None:
            deployment.run_timeout = machine_version.run_timeout
        if update_data is None or update_data.idle_timeout is None:
            deployment.idle_timeout = machine_version.idle_timeout
    
    # Update fields from update_data if provided
    if update_data is not None:
        if update_data.concurrency_limit is not None:
            deployment.concurrency_limit = update_data.concurrency_limit
        if update_data.gpu is not None:
            deployment.gpu = update_data.gpu
        if update_data.run_timeout is not None:
            deployment.run_timeout = update_data.run_timeout
        if update_data.idle_timeout is not None:
            deployment.idle_timeout = update_data.idle_timeout
        if update_data.keep_warm is not None:
            deployment.keep_warm = update_data.keep_warm

    # Check if any deployment-critical parameters have changed
    should_redeploy = (
        (machine_version is not None and machine_version.modal_image_id is not None and original_modal_image_id != machine_version.modal_image_id) or
        original_run_timeout != deployment.run_timeout or
        original_idle_timeout != deployment.idle_timeout or
        original_concurrency_limit != deployment.concurrency_limit
    )


class DeploymentUpdate(BaseModel):
    workflow_version_id: Optional[str] = None
    machine_id: Optional[str] = None
    machine_version_id: Optional[str] = None
    concurrency_limit: Optional[int] = None
    gpu: Optional[GPUType] = None
    run_timeout: Optional[int] = None
    idle_timeout: Optional[int] = None
    keep_warm: Optional[int] = None

async def update_deployment_with_machine(
    deployment: Deployment,
    machine_id: str,
    machine_version: Optional[MachineVersion],
    db: AsyncSession,
    update_data: Optional[DeploymentUpdate] = None,
) -> Deployment:
    """Update deployment with machine and machine version information."""
    # Store original values to check for changes
    original_modal_image_id = deployment.modal_image_id
    original_run_timeout = deployment.run_timeout
    original_idle_timeout = deployment.idle_timeout
    original_concurrency_limit = deployment.concurrency_limit
    original_keep_warm = deployment.keep_warm
    
    deployment.machine_id = machine_id
            
    if machine_version is not None and machine_version.modal_image_id is not None:
        deployment.machine_version_id = machine_version.id
        deployment.modal_image_id = machine_version.modal_image_id
        
        # Only update these fields from machine_version if not provided in update_data
        if update_data is None or update_data.gpu is None:
            deployment.gpu = machine_version.gpu
        if update_data is None or update_data.run_timeout is None:
            deployment.run_timeout = machine_version.run_timeout
        if update_data is None or update_data.idle_timeout is None:
            deployment.idle_timeout = machine_version.idle_timeout
    
    # Update fields from update_data if provided
    if update_data is not None:
        if update_data.concurrency_limit is not None:
            deployment.concurrency_limit = update_data.concurrency_limit
        if update_data.gpu is not None:
            deployment.gpu = update_data.gpu
        if update_data.run_timeout is not None:
            deployment.run_timeout = update_data.run_timeout
        if update_data.idle_timeout is not None:
            deployment.idle_timeout = update_data.idle_timeout
        if update_data.keep_warm is not None:
            deployment.keep_warm = update_data.keep_warm

    # Check if any deployment-critical parameters have changed
    should_redeploy = (
        (machine_version is not None and machine_version.modal_image_id is not None and original_modal_image_id != machine_version.modal_image_id) or
        original_run_timeout != deployment.run_timeout or
        original_idle_timeout != deployment.idle_timeout or
        original_concurrency_limit != deployment.concurrency_limit
    )


    if should_redeploy:
        # We should trigger a redeploy with the final values
        await redeploy_comfy_deploy_runner_if_exists(machine_id, deployment.gpu, deployment)

    # Handle keep_warm changes
    keep_warm_changed = original_keep_warm != deployment.keep_warm
    if keep_warm_changed:
        logger.info(f"Keep warm changed for deployment {deployment.id} to {deployment.keep_warm}")
        try:
            await set_machine_always_on(
                str(deployment.id),
                KeepWarmBody(warm_pool_size=deployment.keep_warm, gpu=GPUType(deployment.gpu)),
            )
        except Exception as e:
            # This is expected to fail if the deployment is not found
            logger.warning(f"Error setting machine always on: {e}", exc_info=True)
                
    return deployment
    if should_redeploy:
        # We should trigger a redeploy with the final values
        await redeploy_comfy_deploy_runner_if_exists(machine_id, deployment.gpu, deployment)

    # Handle keep_warm changes
    keep_warm_changed = original_keep_warm != deployment.keep_warm
    if keep_warm_changed:
        logger.info(f"Keep warm changed for deployment {deployment.id} to {deployment.keep_warm}")
        try:
            await set_machine_always_on(
                str(deployment.id),
                KeepWarmBody(warm_pool_size=deployment.keep_warm, gpu=GPUType(deployment.gpu)),
            )
        except Exception as e:
            # This is expected to fail if the deployment is not found
            logger.warning(f"Error setting machine always on: {e}", exc_info=True)
                
    return deployment

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
            existing_deployment.share_slug = deployment_data.share_slug
            existing_deployment.workflow_version_id = deployment_data.workflow_version_id
            existing_deployment.description = deployment_data.description
            deployment = await update_deployment_with_machine(existing_deployment, machine_id, machine_version, db)
        else:
            # Create new deployment object
            deployment = Deployment(
                id=uuid.uuid4(),
                user_id=user_id,
                org_id=org_id,
                workflow_version_id=deployment_data.workflow_version_id,
                workflow_id=deployment_data.workflow_id,
                environment=deployment_data.environment,
                description=deployment_data.description,
                share_slug=deployment_data.share_slug,
            )
            deployment = await update_deployment_with_machine(deployment, machine_id, machine_version, db)
            db.add(deployment)

        await db.commit()
        await db.refresh(deployment)

        # Convert to dict
        deployment_dict = deployment.to_dict()
        return deployment_dict
    except Exception as e:
        logger.error(f"Error creating deployment: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

@router.patch(
    "/deployment/{deployment_id}",
    response_model=DeploymentModel,
    openapi_extra={
        "x-speakeasy-name-override": "update",
    },
)
async def update_deployment(
    request: Request,
    deployment_id: str,
    deployment_data: DeploymentUpdate,
    db: AsyncSession = Depends(get_db),
):
    try:
        # Get existing deployment
        deployment_query = select(Deployment).where(
            Deployment.id == deployment_id
        ).apply_org_check(request)
            
        result = await db.execute(deployment_query)
        deployment = result.scalar_one_or_none()
        
        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        logger.info(f"Deployment data: {deployment_data}")
        
        machine_version = None
        machine_id = deployment_data.machine_id or deployment.machine_id
    
        if deployment_data.machine_version_id is not None:
            machine_version_query = select(MachineVersion).where(MachineVersion.id == deployment_data.machine_version_id)
            result = await db.execute(machine_version_query)
            machine_version = result.scalar_one_or_none()
            machine_id = machine_version.machine_id
            
        # Get current machine and machine version
        if machine_id:
            machine_query = select(Machine).where(Machine.id == machine_id)
            result = await db.execute(machine_query)
            machine = result.scalar_one_or_none()
            if not machine:
                raise HTTPException(status_code=404, detail="Machine not found")

            if machine.machine_version_id and machine_version is None:
                machine_version_query = select(MachineVersion).where(MachineVersion.id == machine.machine_version_id)
                result = await db.execute(machine_version_query)
                machine_version = result.scalar_one_or_none()

        # Update workflow version if provided
        if deployment_data.workflow_version_id:
            deployment.workflow_version_id = deployment_data.workflow_version_id

        # Update machine-related fields and other deployment settings
        deployment = await update_deployment_with_machine(deployment, machine_id, machine_version, db, deployment_data)

        await db.commit()
        await db.refresh(deployment)

        # Convert to dict
        deployment_dict = deployment.to_dict()
        return deployment_dict

    except Exception as e:
        logger.error(f"Error updating deployment: {e}", exc_info=True)
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
        ).join(Workflow).where(Workflow.deleted == False).order_by(Deployment.updated_at.desc())

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


@router.get(
    "/share/{username}/{slug}",
    response_model=DeploymentShareModel,
)
async def get_share_deployment(
    request: Request,
    username: str,
    slug: str,
    db: AsyncSession = Depends(get_db),
):
    user_data = await get_clerk_data_with_slug(username)
    if user_data["type"] == "none":
        raise HTTPException(status_code=404, detail="User or organization not found")

    # get the deployment with the user id and slug (share_slug)
    deployment_query = (
        select(Deployment)
        .options(
            joinedload(Deployment.workflow).load_only(Workflow.name),
            joinedload(Deployment.version),
        )
        .join(Workflow)
        .where(
            Deployment.share_slug == slug,
            Deployment.environment == "public-share",
            Workflow.deleted == False,
        )
    )
    if user_data["type"] == "user":
        deployment_query = deployment_query.where(Deployment.user_id == user_data["id"])
    elif user_data["type"] == "org":
        deployment_query = deployment_query.where(Deployment.org_id == user_data["id"])

    result = await db.execute(deployment_query)
    deployment = result.scalar_one_or_none()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    # if the deployment is created in org, using userid should fail
    if deployment.org_id and user_data["type"] == "user":
        raise HTTPException(status_code=403, detail="You do not have access to this deployment")

    workflow_api = deployment.version.workflow_api if deployment.version else None
    inputs = get_inputs_from_workflow_api(workflow_api)

    workflow = deployment.version.workflow if deployment.version else None
    outputs = get_outputs_from_workflow(workflow)

    # Just update the deployment with the additional fields
    deployment_dict = deployment.to_dict()
    deployment_dict["input_types"] = inputs
    deployment_dict["output_types"] = outputs

    # FastAPI will automatically filter based on DeploymentModel
    return deployment_dict
