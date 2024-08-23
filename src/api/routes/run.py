import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
import modal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from .utils import select

# from sqlalchemy import select
from api.models import WorkflowRun, Deployment, Machine, WorkflowVersion, Workflow
from api.database import get_db
from typing import Optional, Union, cast
from pydantic import BaseModel, Field
from typing import Dict, Any
from uuid import UUID
import logging
from datetime import datetime
import logfire
from pprint import pprint
import json

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])


@router.get("/run")
async def get_run(request: Request, run_id: str, db: AsyncSession = Depends(get_db)):
    query = select(WorkflowRun).options(joinedload(WorkflowRun.outputs)).where(WorkflowRun.id == run_id).apply_org_check(request)

    result = await db.execute(query)
    run = result.unique().scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = cast(WorkflowRun, run)
    
    # Convert the run to a dictionary and remove the run_log
    run_dict = {k: v for k, v in vars(run).items() if k != 'run_log'}
    
    return run_dict


class WorkflowRunRequest(BaseModel):
    workflow_id: UUID
    workflow_api_json: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    machine_id: Optional[UUID] = None

    webhook: Optional[str] = None
    webhook_intermediate_status: Optional[bool] = False


class DeploymentRunRequest(BaseModel):
    deployment_id: UUID
    inputs: Dict[str, Any] = Field(default_factory=dict)

    webhook: Optional[str] = None
    webhook_intermediate_status: Optional[bool] = False


CreateRunRequest = Union[WorkflowRunRequest, DeploymentRunRequest]


class CreateRunResponse(BaseModel):
    run_id: UUID


class Input(BaseModel):
    prompt_id: str
    workflow_api: Optional[dict] = None
    inputs: Optional[dict]
    workflow_api_raw: dict
    status_endpoint: str
    file_upload_endpoint: str


@router.post("/run", response_model=CreateRunResponse)
async def create_run(
    request: Request, data: CreateRunRequest, db: AsyncSession = Depends(get_db)
):
    if isinstance(data, WorkflowRunRequest):
        pass
    elif isinstance(data, DeploymentRunRequest):
        # Retrieve the deployment and its associated workflow version
        deployment_query = (
            select(Deployment)
            .where(Deployment.id == data.deployment_id)
            .apply_org_check(request)
        )
        deployment_result = await db.execute(deployment_query)
        deployment = deployment_result.scalar_one_or_none()
        deployment = cast(Optional[Deployment], deployment)

        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        # Get the workflow version associated with the deployment
        workflow_version_query = select(WorkflowVersion).where(
            WorkflowVersion.id == deployment.workflow_version_id
        )
        workflow_version_result = await db.execute(workflow_version_query)
        workflow_version = workflow_version_result.scalar_one_or_none()
        workflow_version = cast(Optional[WorkflowVersion], workflow_version)

        if not workflow_version:
            raise HTTPException(
                status_code=404, detail="Workflow version not found for this deployment"
            )

        # Get the machine associated with the deployment
        machine_query = (
            select(Machine)
            .where(Machine.id == deployment.machine_id)
            .apply_org_check(request)
        )
        machine_result = await db.execute(machine_query)
        machine = machine_result.scalar_one_or_none()
        machine = cast(Optional[Machine], machine)

        if not machine:
            raise HTTPException(
                status_code=404, detail="Machine not found for this deployment"
            )

        prompt_id = uuid.uuid4()

        # Create a new run
        new_run = WorkflowRun(
            id=prompt_id,
            workflow_id=deployment.workflow_id,
            workflow_version_id=workflow_version.id,
            workflow_inputs=data.inputs,
            # User
            user_id=request.state.current_user["user_id"],
            org_id=request.state.current_user["org_id"],
            origin="api",
            # Machine
            machine_id=deployment.machine_id,
            machine_type=machine.type,
            gpu=machine.gpu,
            # Webhook
            webhook=data.webhook,
            webhook_intermediate_status=data.webhook_intermediate_status,
        )
        db.add(new_run)
        await db.commit()
        await db.refresh(new_run)

        # params = Input(
        #     prompt_id=str(new_run.id),
        #     workflow_api_raw=workflow_version.workflow_api,
        #     inputs=data.inputs,
        #     status_endpoint="https://deer-light-gull.ngrok-free.app/api/update-run",
        #     file_upload_endpoint="https://deer-light-gull.ngrok-free.app/api/file-upload",
        # )

        logger.info(machine.id)

        ComfyDeployRunner = modal.Cls.lookup(str(machine.id), "ComfyDeployRunner")
        logger.info(ComfyDeployRunner)
        
        with logfire.span("create-run"):
            result = ComfyDeployRunner().run.remote(
                {
                    "prompt_id": str(new_run.id),
                    "workflow_api_raw": workflow_version.workflow_api,
                    "inputs": data.inputs,
                    "status_endpoint": "https://deer-light-gull.ngrok-free.app/api/update-run",
                    "file_upload_endpoint": "https://deer-light-gull.ngrok-free.app/api/file-upload",
                }
            )

        if 'object_id' in result:
            new_run.modal_function_call_id = result.object_id
            await db.commit()
            await db.refresh(new_run)

        # TODO: Implement logic to start the run asynchronously
        # This could involve sending a message to a queue or calling a background task

        return {"run_id": new_run.id}

        # Now you have access to both deployment and its workflow version
        deployment_id = deployment.id
        workflow_version_id = workflow_version.id

        print("fuck you", deployment_id, workflow_version_id)

        return {"run_id": "0d4e1bd3-9c35-45d4-882d-ae008c7fc9e3"}
    else:
        raise HTTPException(status_code=400, detail="Invalid request")