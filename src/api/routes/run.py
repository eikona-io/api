from enum import Enum
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Response
import modal
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from api.routes.internal import send_realtime_update, send_workflow_update
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


class WorkflowRunOrigin(str, Enum):
    MANUAL = "manual"
    API = "api"
    PUBLIC_SHARE = "public-share"


class WorkflowRequestShare(BaseModel):
    execution_mode: Optional[Literal["async", "sync", "sync_first_result"]] = "async"
    inputs: Dict[str, Any] = Field(default_factory=dict)

    webhook: Optional[str] = None
    webhook_intermediate_status: Optional[bool] = False

    origin: Optional[str] = "api"
    batch_number: Optional[int] = None

    batch_input_params: Optional[Dict[str, List[Any]]] = Field(
        default=None,
        example={
            "input_number": [1, 2, 3],
            "input_text": ["apple", "banana", "cherry"],
        },
        description="Optional dictionary of batch input parameters. Keys are input names, values are lists of inputs.",
    )


class WorkflowRunRequest(WorkflowRequestShare):
    workflow_id: UUID
    workflow_api_json: str
    machine_id: Optional[UUID] = None


class WorkflowRunVersionRequest(WorkflowRequestShare):
    workflow_version_id: UUID
    machine_id: Optional[UUID] = None


class DeploymentRunRequest(WorkflowRequestShare):
    deployment_id: UUID


CreateRunRequest = Union[
    WorkflowRunVersionRequest, WorkflowRunRequest, DeploymentRunRequest
]


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
    if (
        data.batch_number is not None
        and data.batch_number > 1
        and data.execution_mode == "sync_first_result"
    ):
        raise HTTPException(
            status_code=400,
            detail="Batch number is not supported for sync_first_result execution mode",
        )

    machine = None
    machine_id = None

    workflow_version_version = None
    workflow_version_id = None

    workflow_id = None
    workflow_api_raw = None

    if isinstance(data, WorkflowRunVersionRequest):
        workflow_version_id = data.workflow_version_id
        machine_id = data.machine_id
    elif isinstance(data, WorkflowRunRequest):
        workflow_api_raw = data.workflow_api_json
        workflow_id = data.workflow_id
        machine_id = data.machine_id
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

        workflow_version_id = deployment.workflow_version_id
        machine_id = deployment.machine_id

    # Get the workflow version associated with the deployment
    if workflow_version_id is not None:
        workflow_version_query = select(WorkflowVersion).where(
            WorkflowVersion.id == workflow_version_id
        )
        workflow_version_result = await db.execute(workflow_version_query)
        workflow_version = workflow_version_result.scalar_one_or_none()
        workflow_version = cast(Optional[WorkflowVersion], workflow_version)

        if not workflow_version:
            raise HTTPException(
                status_code=404, detail="Workflow version not found for this deployment"
            )

        workflow_api_raw = workflow_version.workflow_api
        workflow_id = workflow_version.workflow_id
        workflow_version_version = workflow_version.version

    if machine_id is not None:
        # Get the machine associated with the deployment
        machine_query = (
            select(Machine).where(Machine.id == machine_id).apply_org_check(request)
        )
        machine_result = await db.execute(machine_query)
        machine = machine_result.scalar_one_or_none()
        machine = cast(Optional[Machine], machine)

        if not machine:
            raise HTTPException(
                status_code=404, detail="Machine not found for this deployment"
            )

    if not workflow_api_raw:
        raise HTTPException(status_code=404, detail="Workflow API not found")

    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    async def run(inputs: Dict[str, Any], batch_id: Optional[UUID] = None):
        prompt_id = uuid.uuid4()

        # Create a new run
        new_run = WorkflowRun(
            id=prompt_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
            workflow_inputs=inputs if inputs is not None else data.inputs,
            workflow_api=workflow_api_raw,
            # User
            user_id=request.state.current_user["user_id"],
            org_id=request.state.current_user["org_id"],
            origin=data.origin,
            # Machine
            machine_id=machine.id,
            machine_type=machine.type,
            gpu=machine.gpu,
            # Webhook
            webhook=data.webhook,
            webhook_intermediate_status=data.webhook_intermediate_status,
            batch_id=batch_id,
        )
        db.add(new_run)
        await db.commit()
        await db.refresh(new_run)

        ComfyDeployRunner = modal.Cls.lookup(str(machine.id), "ComfyDeployRunner")

        params = {
            "prompt_id": str(new_run.id),
            "workflow_api_raw": workflow_api_raw,
            "inputs": data.inputs,
            "status_endpoint": os.environ.get("CURRENT_API_URL") + "/api/update-run",
            "file_upload_endpoint": os.environ.get("CURRENT_API_URL")
            + "/api/file-upload",
        }

        # Get the count of runs for this workflow
        run_count_query = select(func.count(WorkflowRun.id)).where(
            WorkflowRun.workflow_id == workflow_id
        )
        result = await db.execute(run_count_query)
        run_count = result.scalar_one()

        new_run_data = new_run.to_dict()
        new_run_data["version"] = {
            "version": workflow_version_version,
        }
        new_run_data["machine"] = {
            "name": machine.name,
        }
        new_run_data["number"] = run_count
        await send_workflow_update(str(new_run.workflow_id), new_run_data)
        await send_realtime_update(str(new_run.id), new_run.to_dict())

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

            first_output_query = (
                select(WorkflowRunOutput)
                .where(WorkflowRunOutput.run_id == new_run.id)
                .order_by(WorkflowRunOutput.created_at.desc())
                .limit(1)
            )

            result = await db.execute(first_output_query)
            output = result.scalar_one_or_none()

            if data.execution_mode == "sync_first_result":
                if output and output.data and isinstance(output.data, dict):
                    images = output.data.get("images", [])
                    for image in images:
                        if isinstance(image, dict):
                            if "url" in image:
                                # Fetch the image/video data
                                async with httpx.AsyncClient() as client:
                                    response = await client.get(image["url"])
                                    if response.status_code == 200:
                                        content_type = response.headers.get(
                                            "content-type"
                                        )
                                        if content_type:
                                            return Response(
                                                content=response.content,
                                                media_type=content_type,
                                            )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="No output found is matching, please check the workflow or disable output_first_result",
                    )
            else:
                return output
        else:
            raise HTTPException(status_code=400, detail="Invalid execution_mode")

    if data.batch_input_params is not None:
        batch_id = uuid.uuid4()

        # Generate a grid of all combinations of input parameters
        import itertools

        # Get all parameter names and their corresponding values
        param_names = list(data.batch_input_params.keys())
        param_values = list(data.batch_input_params.values())

        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # print(combinations)

        async def batch_run():
            results = []
            for combination in combinations:
                # Create a new input dictionary for each combination
                new_inputs = data.inputs.copy()
                for name, value in zip(param_names, combination):
                    new_inputs[name] = value
                    
                print(new_inputs)

                # # Create a new request object with the updated inputs
                new_data = data.model_copy(update={"inputs": new_inputs})

                # Run the workflow with the new inputs
                result = await run(new_data.inputs, batch_id)
                results.append(result)
            return results

        results = await batch_run()

        return {"status": "success", "run_id": str(batch_id)}

    elif data.batch_number is not None and data.batch_number > 1:

        async def batch_run():
            results = []
            for _ in range(data.batch_number):
                result = await run()
                results.append(result)
            return results

        await batch_run()

        return {"status": "success"}
    else:
        return await run()