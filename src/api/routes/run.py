import asyncio
import datetime
import os
from pprint import pprint
from urllib.parse import urljoin
import uuid

from api.sqlmodels import WorkflowRunWebhookBody, WorkflowRunWebhookResponse
from .types import (
    CreateRunBatchResponse,
    CreateRunRequest,
    CreateRunResponse,
    DeploymentRunRequest,
    RunStream,
    WorkflowRequestShare,
    WorkflowRunModel,
    WorkflowRunNativeOutputModel,
    WorkflowRunOutputModel,
    WorkflowRunRequest,
    WorkflowRunVersionRequest,
)
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Body
from fastapi.responses import StreamingResponse
import modal
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from fastapi import BackgroundTasks
from .internal import insert_to_clickhouse, send_realtime_update, send_workflow_update
from .utils import (
    ensure_run_timeout,
    generate_temporary_token,
    get_user_settings,
    post_process_output_data,
    post_process_outputs,
    select,
)
from clickhouse_connect.driver.asyncclient import AsyncClient

import datetime as dt

# from sqlalchemy import select
from api.models import (
    WorkflowRun,
    Deployment,
    Machine,
    WorkflowRunOutput,
    WorkflowRunWithExtra,
    WorkflowVersion,
    Workflow,
)
from api.database import get_db, get_clickhouse_client, get_db_context
from typing import Optional, Union, cast
from typing import Dict, Any
from uuid import UUID, uuid4
import logging
import logfire
import json
import httpx
from typing import Optional, List
from uuid import UUID
import base64

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Run"])
webhook_router = APIRouter(tags=["Callbacks"])


@webhook_router.post(
    "{$request.body#/webhook}",
    response_model=WorkflowRunWebhookResponse,
    summary="Receive run status updates via webhook",
    description="This endpoint is called by the workflow runner to update the status of a run.",
)
async def run_update_webhook(
    body: WorkflowRunWebhookBody = Body(description="The updated run information"),
):
    # Implement the webhook update logic here
    pass


@router.get(
    "/run/{run_id}",
    response_model=WorkflowRunModel,
    openapi_extra={
        "x-speakeasy-name-override": "get",
    },
)
@router.get("/run", response_model=WorkflowRunModel, include_in_schema=False)
async def get_run(request: Request, run_id: str, db: AsyncSession = Depends(get_db)):
    query = (
        select(WorkflowRunWithExtra)
        .options(joinedload(WorkflowRun.outputs))
        .where(WorkflowRun.id == run_id)
        .apply_org_check(request)
    )

    result = await db.execute(query)
    run = result.unique().scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    run = cast(WorkflowRun, run)

    user_settings = await get_user_settings(request, db)
    ensure_run_timeout(run)
    post_process_outputs(run.outputs, user_settings)

    # Convert the run to a dictionary and remove the run_log
    run_dict = {k: v for k, v in vars(run).items() if k != "run_log"}

    return run_dict


def get_comfy_deploy_runner(machine_id: str, gpu: str):
    ComfyDeployRunner = modal.Cls.lookup(str(machine_id), "ComfyDeployRunner")
    return ComfyDeployRunner.with_options(gpu=gpu if gpu != "CPU" else None)(gpu=gpu)


@router.post(
    "/run",
    response_model=Union[
        CreateRunResponse,
        CreateRunBatchResponse,
        WorkflowRunOutputModel,
        WorkflowRunNativeOutputModel,
    ],
    summary="Run a workflow",
    description="Create a new workflow run with the given parameters. This function sets up the run and initiates the execution process. For callback information, see [Callbacks](#tag/callbacks/POST/\{callback_url\}).",
    callbacks=webhook_router.routes,
    include_in_schema=False,
    # openapi_extra={
    #     "x-speakeasy-name-override": "create",
    # },
)
async def create_run_all(
    request: Request,
    data: CreateRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    client: AsyncClient = Depends(get_clickhouse_client),
):
    return await _create_run(request, data, background_tasks, db, client)


@router.post(
    "/run/queue",
    response_model=CreateRunResponse,
    summary="Queue a workflow",
    description="Create a new workflow run with the given parameters. This function sets up the run and initiates the execution process. For callback information, see [Callbacks](#tag/callbacks/POST/\{callback_url\}).",
    callbacks=webhook_router.routes,
    # include_in_schema=False,
    openapi_extra={
        "x-speakeasy-name-override": "queue",
    },
)
async def create_run_queue(
    request: Request,
    data: CreateRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    client: AsyncClient = Depends(get_clickhouse_client),
):
    return await _create_run(request, data, background_tasks, db, client)


@router.post(
    "/run/sync",
    response_model=List[WorkflowRunOutputModel],
    summary="Run a workflow in sync",
    description="Create a new workflow run with the given parameters. This function sets up the run and initiates the execution process. For callback information, see [Callbacks](#tag/callbacks/POST/\{callback_url\}).",
    callbacks=webhook_router.routes,
    # include_in_schema=False,
    openapi_extra={
        "x-speakeasy-name-override": "sync",
    },
)
async def create_run_sync(
    request: Request,
    data: CreateRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    client: AsyncClient = Depends(get_clickhouse_client),
):
    data.execution_mode = "sync"
    return await _create_run(request, data, background_tasks, db, client)


@router.post(
    "/run/stream",
    response_model=RunStream,
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Stream of workflow run events",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "$ref": "#/components/schemas/RunStream",
                    },
                },
            },
        }
    },
    summary="Run a workflow in stream",
    description="Create a new workflow run with the given parameters. This function sets up the run and initiates the execution process. For callback information, see [Callbacks](#tag/callbacks/POST/\{callback_url\}).",
    callbacks=webhook_router.routes,
    # include_in_schema=False,
    openapi_extra={
        "x-speakeasy-name-override": "stream",
    },
)
async def create_run_stream(
    request: Request,
    data: CreateRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    client: AsyncClient = Depends(get_clickhouse_client),
) -> StreamingResponse:
    data.execution_mode = "stream"
    return await _create_run(request, data, background_tasks, db, client)


# @router.post(
#     "/run/workflow",
#     response_model=Union[
#         CreateRunResponse,
#         CreateRunBatchResponse,
#         WorkflowRunOutputModel,
#         WorkflowRunNativeOutputModel,
#     ],
#     summary="Run comfyui workflow",
#     description="Create a new workflow run with the given parameters. This function sets up the run and initiates the execution process. For callback information, see [Callbacks](#tag/callbacks/POST/\{callback_url\}).",
#     callbacks=webhook_router.routes,
# )
# async def create_run_workflow(
#     request: Request,
#     data: WorkflowRunRequest,
#     background_tasks: BackgroundTasks,
#     db: AsyncSession = Depends(get_db),
#     client: AsyncClient = Depends(get_clickhouse_client),
# ):
#     return await _create_run(request, data, background_tasks, db, client)


# @router.post(
#     "/run",
#     response_model=Union[
#         CreateRunResponse,
#         CreateRunBatchResponse,
#         WorkflowRunOutputModel,
#         WorkflowRunNativeOutputModel,
#     ],
#     summary="Run workflow",
#     description="Create a new workflow run with the given parameters. This function sets up the run and initiates the execution process. For callback information, see [Callbacks](#tag/callbacks/POST/\{callback_url\}).",
#     callbacks=webhook_router.routes,
#     openapi_extra={
#         "x-speakeasy-name-override": "create",
#     },
# )
# async def create_run(
#     request: Request,
#     data: WorkflowRequestShare,
#     background_tasks: BackgroundTasks,
#     db: AsyncSession = Depends(get_db),
#     client: AsyncClient = Depends(get_clickhouse_client),
# ):
#     data = DeploymentRunRequest(deployment_id=deployment_id, **data.model_dump())
#     return await _create_run(request, data, background_tasks, db, client)


async def _create_run(
    request: Request,
    data: CreateRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    client: AsyncClient = Depends(get_clickhouse_client),
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

    org_id = (
        request.state.current_user["org_id"]
        if "org_id" in request.state.current_user
        else None
    )

    is_native_run = data.is_native_run

    if isinstance(data, WorkflowRunVersionRequest):
        workflow_version_id = data.workflow_version_id
        machine_id = data.machine_id
    elif isinstance(data, WorkflowRunRequest):
        workflow_api_raw = data.workflow_api_json
        workflow_id = data.workflow_id
        machine_id = data.machine_id
        workflow = data.workflow
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
        workflow = workflow_version.workflow

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

    async def run(inputs: Dict[str, Any] = None, batch_id: Optional[UUID] = None):
        prompt_id = uuid.uuid4()
        user_id = request.state.current_user["user_id"]


        # Create a new run
        new_run = WorkflowRun(
            id=prompt_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
            workflow_inputs=inputs if inputs is not None else data.inputs,
            workflow_api=workflow_api_raw,
            # User
            user_id=user_id,
            org_id=org_id,
            origin=data.origin,
            # Machine
            machine_id=machine_id,
            machine_type=machine.type,
            gpu=machine.gpu,
            # Webhook
            webhook=data.webhook,
            webhook_intermediate_status=data.webhook_intermediate_status,
            batch_id=batch_id,
        )

        if is_native_run:
            new_run.queued_at = dt.datetime.now(dt.UTC)
            new_run.started_at = dt.datetime.now(dt.UTC)

        db.add(new_run)
        await db.commit()
        await db.refresh(new_run)

        params = {
            "prompt_id": str(new_run.id),
            "workflow_api_raw": workflow_api_raw,
            "inputs": inputs,
            "status_endpoint": os.environ.get("CURRENT_API_URL") + "/api/update-run",
            "file_upload_endpoint": os.environ.get("CURRENT_API_URL")
            + "/api/file-upload",
            "workflow": workflow,
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
        background_tasks.add_task(
            send_workflow_update, str(new_run.workflow_id), new_run_data
        )
        background_tasks.add_task(
            send_realtime_update, str(new_run.id), new_run.to_dict()
        )

        # Sending to clickhouse
        progress_data = [
            (
                user_id,
                org_id,
                machine_id,
                None, # gpu_event_id
                workflow_id,
                workflow_version_id,
                new_run.id,
                dt.datetime.now(dt.UTC),
                "queued",
                0,
                "",
            )
        ]

        background_tasks.add_task(
            insert_to_clickhouse, client, "workflow_events", progress_data
        )

        token = generate_temporary_token(request.state.current_user["user_id"], org_id)
        # logger.info(token)
        # logger.info("machine type " + machine.type)

        # return the params for the native run
        if is_native_run:
            return {
                **params,
                "cd_token": token,
            }

        if data.execution_mode == "async":
            match machine.type:
                case "comfy-deploy-serverless":
                    # print("shit", str(machine_id))
                    ComfyDeployRunner = get_comfy_deploy_runner(machine_id, machine.gpu)
                    with logfire.span("spawn-run"):
                        result = ComfyDeployRunner.run.spawn(params)
                        new_run.modal_function_call_id = result.object_id
                # For runpod there will be a problem with the auth token cause v2 endpoint requires a token
                case "runpod-serverless":
                    if not machine.auth_token:
                        raise HTTPException(
                            status_code=400, detail="Machine auth token not found"
                        )

                    async with httpx.AsyncClient() as _client:
                        try:
                            payload = {"input": params}
                            response = await _client.post(
                                f"{machine.endpoint}/run",
                                json=payload,
                                headers={
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {machine.auth_token}",
                                },
                            )
                            response.raise_for_status()
                        except httpx.HTTPStatusError as e:
                            raise HTTPException(
                                status_code=e.response.status_code,
                                detail=f"Error creating run: {e.response.text}",
                            )

                    # Update the run with the RunPod job ID if available
                    runpod_response = response.json()
                case "classic":
                    # comfyui_endpoint = f"{machine.endpoint}/comfyui-deploy/run"
                    comfyui_endpoint = urljoin(machine.endpoint, "comfyui-deploy/run")

                    headers = {"Content-Type": "application/json"}
                    # if machine.auth_token:
                    # headers["Authorization"] = f"Bearer {machine.auth_token}"
                    if machine.auth_token:
                        # Use Basic Authentication
                        credentials = base64.b64encode(
                            machine.auth_token.encode()
                        ).decode()
                        headers["Authorization"] = f"Basic {credentials}"

                    # print(headers)

                    async with httpx.AsyncClient() as _client:
                        try:
                            response = await _client.post(
                                comfyui_endpoint,
                                json={
                                    **params,
                                    "cd_token": token,
                                },
                                headers=headers,
                            )
                            response.raise_for_status()
                        except httpx.HTTPStatusError as e:
                            error_message = f"Error creating run: {e.response.status_code} {e.response.reason_phrase}"
                            try:
                                result = response.json()
                                if "node_errors" in result:
                                    error_message += f" {result['node_errors']}"
                            except json.JSONDecodeError:
                                pass
                            raise HTTPException(
                                status_code=e.response.status_code, detail=error_message
                            )
                case _:
                    raise HTTPException(status_code=400, detail="Invalid machine type")

            await db.commit()
            await db.refresh(new_run)

            return {"run_id": new_run.id}
        elif data.execution_mode in ["sync", "sync_first_result"]:
            with logfire.span("run-sync"):
                ComfyDeployRunner = get_comfy_deploy_runner(machine_id, machine.gpu)
                result = await ComfyDeployRunner.run.remote.aio(params)

            if data.execution_mode == "sync_first_result":
                first_output_query = (
                    select(WorkflowRunOutput)
                    .where(WorkflowRunOutput.run_id == new_run.id)
                    .order_by(WorkflowRunOutput.created_at.desc())
                    .limit(1)
                )

                result = await db.execute(first_output_query)
                output = result.scalar_one_or_none()

                user_settings = await get_user_settings(request, db)

                post_process_outputs([output], user_settings)

                if data.execution_mode == "sync_first_result":
                    if output and output.data and isinstance(output.data, dict):
                        images = output.data.get("images", [])
                        for image in images:
                            if isinstance(image, dict):
                                if "url" in image:
                                    # Fetch the image/video data
                                    async with httpx.AsyncClient() as _client:
                                        response = await _client.get(image["url"])
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
                output_query = (
                    select(WorkflowRunOutput)
                    .where(WorkflowRunOutput.run_id == new_run.id)
                    .order_by(WorkflowRunOutput.created_at.desc())
                )

                result = await db.execute(output_query)
                outputs = result.scalars().all()

                user_settings = await get_user_settings(request, db)
                post_process_outputs(outputs, user_settings)

                return [output.to_dict() for output in outputs]
        elif data.execution_mode == "stream":
            ComfyDeployRunner = get_comfy_deploy_runner(machine_id, machine.gpu)
            user_settings = await get_user_settings(request, db)

            async def wrapped_generator():
                yield f"event: event_update\ndata: {json.dumps({'event': 'queuing'})}\n\n"
                try:
                    with logfire.span("stream-run"):
                        async for event in ComfyDeployRunner.streaming.remote_gen.aio(
                            input=params
                        ):
                            if isinstance(event, (str, bytes)):
                                # Convert bytes to string if necessary
                                event_str = (
                                    event.decode("utf-8")
                                    if isinstance(event, bytes)
                                    else event
                                )
                                lines = event_str.strip().split("\n")
                                event_type = None
                                event_data = None
                                for line in lines:
                                    if line.startswith("event:"):
                                        event_type = line.split(":", 1)[1].strip()
                                    elif line.startswith("data:"):
                                        event_data = line.split(":", 1)[1].strip()

                                # logger.info(event_type)
                                # logger.info(lines)

                                if event_type == "event_update" and event_data:
                                    try:
                                        data = json.loads(event_data)
                                        if data.get("event") == "function_call_id":
                                            new_run.modal_function_call_id = data.get(
                                                "data"
                                            )
                                            await db.commit()
                                        if data.get("event") == "executed":
                                            logger.info(
                                                data.get("data", {}).get("output")
                                            )
                                            post_process_output_data(
                                                data.get("data", {}).get("output"),
                                                user_settings,
                                            )
                                            new_event_data = {
                                                "event": "executed",
                                                "data": data.get("data", {}),
                                            }
                                            new_event = f"event: event_update\ndata: {json.dumps(new_event_data)}\n\n"
                                            yield new_event
                                            continue

                                    except json.JSONDecodeError:
                                        pass  # Invalid JSON, ignore

                            # logger.info(event)
                            yield event

                except Exception as e:
                    print(e)

            return StreamingResponse(
                wrapped_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
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

        return {"status": "success", "batch_id": str(batch_id)}

    elif data.batch_number is not None and data.batch_number > 1:
        batch_id = uuid.uuid4()

        async def batch_run():
            results = []
            for _ in range(data.batch_number):
                result = await run(None, batch_id)
                results.append(result)
            return results

        await batch_run()

        return {"status": "success", "batch_id": str(batch_id)}
    else:
        return await run(
            data.inputs,
        )
