import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Response
import modal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from .utils import select

# from sqlalchemy import select
from api.models import (
    WorkflowRun,
    Deployment,
    Machine,
    WorkflowRunOutput,
    WorkflowVersion,
    Workflow,
)
from api.database import get_db
from typing import Literal, Optional, Union, cast
from pydantic import BaseModel, Field
from typing import Dict, Any
from uuid import UUID
import logging
from datetime import datetime
import logfire
from pprint import pprint
import json
import io
import httpx
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)

router = APIRouter(tags=["run"])


class WorkflowRunOutputModel(BaseModel):
    id: UUID
    run_id: UUID
    data: Optional[Dict[str, Any]]
    node_meta: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

class WorkflowRunModel(BaseModel):
    id: UUID
    workflow_version_id: Optional[UUID]
    workflow_inputs: Optional[Dict[str, Any]]
    workflow_id: UUID
    workflow_api: Optional[Dict[str, Any]]
    machine_id: Optional[UUID]
    origin: str
    status: str
    ended_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    queued_at: Optional[datetime]
    started_at: Optional[datetime]
    gpu_event_id: Optional[str]
    gpu: Optional[str]
    machine_version: Optional[str]
    machine_type: Optional[str]
    modal_function_call_id: Optional[str]
    user_id: Optional[str]
    org_id: Optional[str]
    live_status: Optional[str]
    progress: float = Field(default=0)
    is_realtime: bool = Field(default=False)
    webhook: Optional[str]
    webhook_status: Optional[str]
    webhook_intermediate_status: bool = Field(default=False)
    outputs: List[WorkflowRunOutputModel] = []

    class Config:
        orm_mode = True


@router.get("/run/{run_id}", response_model=WorkflowRunModel)
@router.get("/run", deprecated=True, response_model=WorkflowRunModel)
async def get_run(request: Request, run_id: str, db: AsyncSession = Depends(get_db)):
    query = (
        select(WorkflowRun)
        .options(joinedload(WorkflowRun.outputs))
        .where(WorkflowRun.id == run_id)
        .apply_org_check(request)
    )

    result = await db.execute(query)
    run = result.unique().scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    run = cast(WorkflowRun, run)

    # Convert the run to a dictionary and remove the run_log
    run_dict = {k: v for k, v in vars(run).items() if k != "run_log"}

    return run_dict


class WorkflowRunRequest(BaseModel):
    workflow_id: UUID
    workflow_api_json: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    machine_id: Optional[UUID] = None

    execution_mode: Literal["async", "sync", "sync_first_result"] = "async"

    webhook: Optional[str] = None
    webhook_intermediate_status: Optional[bool] = False


class DeploymentRunRequest(BaseModel):
    deployment_id: UUID
    inputs: Dict[str, Any] = Field(default_factory=dict)
    
    execution_mode: Literal["async", "sync", "sync_first_result"] = "async"
    
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

class WorkflowRunOutputModel(BaseModel):
    id: UUID
    run_id: UUID
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    type: Optional[str] = None
    node_id: Optional[str] = None

    class Config:
        from_attributes = True


@router.post("/run", response_model=Union[CreateRunResponse, WorkflowRunOutputModel])
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

        params = {
            "prompt_id": str(new_run.id),
            "workflow_api_raw": workflow_version.workflow_api,
            "inputs": data.inputs,
            "status_endpoint": os.environ.get("LEGACY_API_URL") + "/api/update-run",
            "file_upload_endpoint": os.environ.get("LEGACY_API_URL")
            + "/api/file-upload",
        }

        if data.execution_mode == "async":
            with logfire.span("spawn-run"):
                result = ComfyDeployRunner().run.spawn(params)

            new_run.modal_function_call_id = result.object_id
            await db.commit()
            await db.refresh(new_run)

            return {"run_id": new_run.id}
        elif data.execution_mode in ["sync", "sync_first_result"]:
            with logfire.span("run-sync"):
                result = await ComfyDeployRunner().run.remote.aio(params)

            first_output_query = select(WorkflowRunOutput).where(
                WorkflowRunOutput.run_id == new_run.id
            ).order_by(WorkflowRunOutput.created_at.desc()).limit(1)
            
            result = await db.execute(first_output_query)
            output = result.scalar_one_or_none()
            
            if data.execution_mode == "sync_first_result":
                if output and output.data and isinstance(output.data, dict):
                    images = output.data.get('images', [])
                    for image in images:
                        if isinstance(image, dict):
                            if 'url' in image:
                                # Fetch the image/video data
                                async with httpx.AsyncClient() as client:
                                    response = await client.get(image['url'])
                                    if response.status_code == 200:
                                        content_type = response.headers.get('content-type')
                                        if content_type:
                                            return Response(content=response.content, media_type=content_type)
                else:
                    raise HTTPException(status_code=400, detail="No output found is matching, please check the workflow or disable output_first_result")
            else:
                return output
        else:
            raise HTTPException(status_code=400, detail="Invalid execution_mode")