import json
import os
from urllib.parse import urlparse, quote
from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    BackgroundTasks,
    Response,
    Request,
)
import asyncio
import clickhouse_connect
from uuid import uuid4
from datetime import datetime, timezone
from fastapi.responses import RedirectResponse
from pydantic import UUID4, BaseModel, Field
from typing import Optional, Any, cast
from datetime import datetime
from enum import Enum
import boto3
from botocore.exceptions import ClientError
from clickhouse_connect.driver.asyncclient import AsyncClient
from botocore.config import Config
import aiohttp
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from pprint import pprint

from .utils import (
    async_lru_cache,
    generate_presigned_url,
    get_user_settings,
    post_process_outputs,
    retry_fetch,
    select,
    send_realtime_update,
    send_workflow_update,
)
from sqlalchemy import update, case

from api.database import AsyncSessionLocal, get_clickhouse_client, get_db
from api.models import Machine, UserSettings, Workflow, WorkflowRun, WorkflowRunOutput

from fastapi import Query
from datetime import datetime, timezone
import datetime as dt

from fastapi import Depends

router = APIRouter()


async def send_webhook(workflow_run, updatedAt, run_id):
    url = workflow_run.webhook
    extra_headers = {}  # You may want to define this based on your requirements

    payload = {
        "status": workflow_run.status,
        "live_status": workflow_run.live_status,
        "progress": workflow_run.progress,
        "run_id": workflow_run.id,
        "outputs": workflow_run.data,  # Assuming 'data' contains the outputs
    }

    headers = {"Content-Type": "application/json", **extra_headers}

    options = {"method": "POST", "headers": headers, "json": payload}

    try:
        response = await retry_fetch(url, options)
        if response.ok:
            return {"status": "success", "message": "Webhook sent successfully"}
        else:
            return {
                "status": "error",
                "message": f"Webhook failed with status {response.status}",
            }
    except Exception as error:
        return {"status": "error", "message": str(error)}


class WorkflowRunStatus(str, Enum):
    NOT_STARTED = "not-started"
    QUEUED = "queued"
    STARTED = "started"
    RUNNING = "running"
    UPLOADING = "uploading"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class UpdateRunBody(BaseModel):
    run_id: str
    status: Optional[WorkflowRunStatus] = None
    time: Optional[datetime] = None
    output_data: Optional[Any] = None
    node_meta: Optional[Any] = None
    log_data: Optional[Any] = None
    logs: Optional[Any] = None
    ws_event: Optional[Any] = None
    live_status: Optional[str] = None
    progress: Optional[float] = None
    modal_function_call_id: Optional[str] = None


router = APIRouter()


async def insert_to_clickhouse(client: AsyncClient, table: str, data: list):
    await client.insert(table=table, data=data)


# async def insert_to_clickhouse_multi(client: AsyncClient, table: str, data: list):
#     # Prepare the data for batch insert
#     columns = list(data[0].keys())
#     values = [list(item.values()) for item in data]

#     # Perform batch insert
#     result = await client.insert(table=table, data=values, column_names=columns)
#     print("result", result)


@async_lru_cache(maxsize=1000)
async def get_cached_workflow_run(run_id: str, db: AsyncSession):
    existing_run = await db.execute(select(WorkflowRun).where(WorkflowRun.id == run_id))
    workflow_run = existing_run.scalar_one_or_none()
    workflow_run = cast(WorkflowRun, workflow_run)
    return workflow_run


@router.post("/update-run", include_in_schema=False)
async def update_run(
    request: Request,
    body: UpdateRunBody,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    client: AsyncClient = Depends(get_clickhouse_client),
):
    # updated_at = datetime.utcnow()
    updated_at = dt.datetime.now(dt.UTC)

    # Ensure request.time is timezone-aware
    # fixed_time = request.time.replace(tzinfo=timezone.utc) if request.time and request.time.tzinfo is None else request.time
    fixed_time = updated_at

    if body.ws_event is not None:
        # print("body.ws_event", body.ws_event)
        # Get the workflow run
        # print("body.run_id", body.run_id)
        workflow_run = await get_cached_workflow_run(body.run_id, db)
        # print("workflow_run", workflow_run)

        log_data = [
            (
                uuid4(),
                body.run_id,
                workflow_run.workflow_id,
                workflow_run.machine_id,
                updated_at,
                "ws_event",
                json.dumps(body.ws_event),
            )
        ]
        # Add ClickHouse insert to background tasks
        background_tasks.add_task(insert_to_clickhouse, client, "log_entries", log_data)
        return {"status": "success"}

    if body.logs is not None:
        # Get the workflow run
        workflow_run = await get_cached_workflow_run(body.run_id, db)

        if not workflow_run:
            raise HTTPException(status_code=404, detail="WorkflowRun not found")

        # Prepare data for ClickHouse insert
        log_data = []
        for log_entry in body.logs:
            data = (
                uuid4(),
                body.run_id,
                workflow_run.workflow_id,
                workflow_run.machine_id,
                datetime.fromtimestamp(log_entry["timestamp"], tz=timezone.utc),
                "info",
                log_entry["logs"],
            )
            print("data", log_entry["logs"])
            log_data.append(data)
            
        background_tasks.add_task(insert_to_clickhouse, client, "log_entries", log_data)
        
        return {"status": "success"}

    # Updating the progress
    if body.live_status is not None and body.progress is not None:
        # Updating the workflow run table with live_status, progress, and updated_at
        update_stmt = (
            update(WorkflowRun)
            .where(WorkflowRun.id == body.run_id)
            .values(
                live_status=body.live_status,
                progress=body.progress,
                updated_at=updated_at,
            )
            .returning(WorkflowRun)
        )
        result = await db.execute(update_stmt)
        await db.commit()

        workflow_run = result.scalar_one()
        workflow_run = cast(WorkflowRun, workflow_run)
        await db.refresh(workflow_run)

        # Sending a real-time update to connected clients
        background_tasks.add_task(
            send_workflow_update, str(workflow_run.workflow_id), workflow_run.to_dict()
        )
        background_tasks.add_task(
            send_realtime_update, str(workflow_run.id), workflow_run.to_dict()
        )

        # Sending to clickhouse
        progress_data = [
            (
                workflow_run.user_id,
                workflow_run.org_id,
                workflow_run.machine_id,
                # container_id as gpu_event_id
                workflow_run.workflow_id,
                workflow_run.workflow_version_id,
                body.run_id,
                updated_at,
                body.progress,
                body.live_status,
                body.status or "executing",
            )
        ]

        background_tasks.add_task(
            insert_to_clickhouse, client, "progress_updates", progress_data
        )

        if workflow_run.webhook and workflow_run.webhook_intermediate_status:
            background_tasks.add_task(
                send_webhook, workflow_run, updated_at, workflow_run.id
            )

        return {"status": "success"}

    if body.log_data is not None:
        # Cause all the logs will be sent to clickhouse now.
        return {"status": "success"}
        # update_stmt = (
        #     update(WorkflowRun)
        #     .where(WorkflowRun.id == request.run_id)
        #     .values(run_log=request.log_data, updated_at=updated_at)
        # )
        # await db.execute(update_stmt)
        # await db.commit()
        # return {"status": "success"}

    if body.status == "started" and fixed_time is not None:
        update_stmt = (
            update(WorkflowRun)
            .where(WorkflowRun.id == body.run_id)
            .values(started_at=fixed_time, updated_at=updated_at)
        )
        await db.execute(update_stmt)
        await db.commit()

    if body.status == "queued" and fixed_time is not None:
        # Ensure request.time is UTC-aware
        update_stmt = (
            update(WorkflowRun)
            .where(WorkflowRun.id == body.run_id)
            .values(queued_at=fixed_time, updated_at=updated_at)
        )
        await db.execute(update_stmt)
        await db.commit()
        # return {"status": "success"}

    ended = body.status in ["success", "failed", "timeout", "cancelled"]
    if body.output_data is not None:
        # Sending to postgres
        newOutput = WorkflowRunOutput(
            id=uuid4(),  # Add this line to generate a new UUID for the primary key
            created_at=updated_at,
            updated_at=updated_at,
            run_id=body.run_id,
            data=body.output_data,
            node_meta=body.node_meta,
        )
        db.add(newOutput)
        await db.commit()
        await db.refresh(newOutput)

        # TODO: Send to upload to clickhouse
        output_data = [
            (
                uuid4(),
                body.run_id,
                workflow_run.workflow_id,
                workflow_run.machine_id,
                updated_at,
                body.progress,
                body.live_status,
                body.status,
                # json.dumps(request.output_data),
                # json.dumps(request.node_meta),
            )
        ]
        background_tasks.add_task(insert_to_clickhouse, client, "progress_updates", output_data)

    elif body.status is not None:
        # Get existing run
        existing_run = await db.execute(
            select(WorkflowRun).where(WorkflowRun.id == body.run_id)
        )
        workflow_run = existing_run.scalar_one_or_none()
        workflow_run = cast(WorkflowRun, workflow_run)

        if not workflow_run:
            raise HTTPException(status_code=404, detail="WorkflowRun not found")

        # If the run is already cancelled, don't update it
        if body.status == "cancelled":
            return {"status": "success"}

        # Update the run status
        update_data = {"status": body.status, "updated_at": updated_at}
        if ended and fixed_time is not None:
            update_data["ended_at"] = fixed_time

        # Sending to clickhouse
        progress_data = [
            (
                uuid4(),
                body.run_id,
                workflow_run.workflow_id,
                workflow_run.machine_id,
                updated_at,
                body.progress,
                body.live_status,
                body.status,
            )
        ]

        background_tasks.add_task(
            insert_to_clickhouse, client, "progress_updates", progress_data
        )

        update_values = {
            "status": body.status,
            "ended_at": updated_at if ended else None,
            "updated_at": updated_at,
        }

        # Add modal_function_call_id if it's provided and the existing value is empty
        if body.modal_function_call_id:
            update_values["modal_function_call_id"] = body.modal_function_call_id

        # print("modal_function_call_id", request.modal_function_call_id)

        update_stmt = (
            update(WorkflowRun)
            .where(WorkflowRun.id == body.run_id)
            .values(**update_values)
            .returning(WorkflowRun)
        )
        result = await db.execute(update_stmt)
        await db.commit()
        workflow_run = result.scalar_one_or_none()
        workflow_run = cast(WorkflowRun, workflow_run)
        await db.refresh(workflow_run)

        # Get all outputs for the workflow run
        outputs_query = select(WorkflowRunOutput).where(
            WorkflowRunOutput.run_id == body.run_id
        )
        outputs_result = await db.execute(outputs_query)
        outputs = outputs_result.scalars().all()

        user_settings = await get_user_settings(request, db)
        if outputs:
            post_process_outputs(outputs, user_settings)
        # Instead of setting outputs directly, create a new dictionary with all the data
        workflow_run_data = workflow_run.to_dict()
        workflow_run_data["outputs"] = [output.to_dict() for output in outputs]
        # workflow_run_data["status"] = request.status.value

        # Move send_workflow_update to background task
        background_tasks.add_task(
            send_workflow_update, str(workflow_run.workflow_id), workflow_run_data
        )
        background_tasks.add_task(
            send_realtime_update, str(workflow_run.id), workflow_run_data
        )

        if workflow_run.webhook:
            background_tasks.add_task(
                send_webhook, workflow_run, updated_at, workflow_run.id
            )

        return {"status": "success"}

    return {"status": "success"}


@router.post("/gpu_event", include_in_schema=False)
async def create_gpu_event(request: Request, data: Any = Body(...)):
    legacy_api_url = os.getenv("LEGACY_API_URL", "").rstrip("/")
    new_url = f"{legacy_api_url}/api/gpu_event"

    # Get headers from the incoming request
    headers = dict(request.headers)
    # Remove host header as it will be set by aiohttp
    headers.pop("host", None)

    async with aiohttp.ClientSession() as session:
        async with session.post(new_url, json=data, headers=headers) as response:
            content = await response.read()
            if response.status >= 400:
                raise HTTPException(
                    status_code=response.status, detail=content.decode()
                )
            return Response(
                content=content,
                status_code=response.status,
                headers=dict(response.headers),
            )


@router.post("/machine-built", include_in_schema=False)
async def machine_built(request: Request, data: Any = Body(...)):
    legacy_api_url = os.getenv("LEGACY_API_URL", "").rstrip("/")
    new_url = f"{legacy_api_url}/api/machine-built"

    # Get headers from the incoming request
    headers = dict(request.headers)
    # Remove host header as it will be set by aiohttp
    headers.pop("host", None)

    async with aiohttp.ClientSession() as session:
        async with session.post(new_url, json=data, headers=headers) as response:
            content = await response.read()
            return Response(
                content=content,
                status_code=response.status,
                headers=dict(response.headers),
            )


@router.post("/fal-webhook", include_in_schema=False)
async def fal_webhook(request: Request, data: Any = Body(...)):
    legacy_api_url = os.getenv("LEGACY_API_URL", "").rstrip("/")
    new_url = f"{legacy_api_url}/api/fal-webhook"

    # Get headers from the incoming request
    headers = dict(request.headers)
    # Remove host header as it will be set by aiohttp
    headers.pop("host", None)

    async with aiohttp.ClientSession() as session:
        async with session.post(new_url, json=data, headers=headers) as response:
            content = await response.read()
            return Response(
                content=content,
                status_code=response.status,
                headers=dict(response.headers),
            )


@router.get("/file-upload", include_in_schema=False)
async def get_file_upload_url(
    request: Request,
    file_name: str = Query(..., description="Name of the file to upload"),
    run_id: UUID4 = Query(..., description="UUID of the run"),
    size: Optional[int] = Query(None, description="Size of the file in bytes"),
    type: str = Query(..., description="Type of the file"),
    public: bool = Query(
        True, description="Whether to make the file publicly accessible"
    ),
    db: AsyncSession = Depends(get_db),
):
    try:
        user_settings = await get_user_settings(request, db)

        bucket = os.getenv("SPACES_BUCKET_V2")
        region = os.getenv("SPACES_REGION_V2")
        access_key = os.getenv("SPACES_KEY_V2")
        secret_key = os.getenv("SPACES_SECRET_V2")

        if user_settings is not None:
            if user_settings.output_visibility == "private":
                public = False

            if user_settings.custom_output_bucket:
                bucket = user_settings.s3_bucket_name
                region = user_settings.s3_region
                access_key = user_settings.s3_access_key_id
                secret_key = user_settings.s3_secret_access_key

        # Generate the object key
        object_key = f"outputs/runs/{run_id}/{file_name}"

        # Encode the object key for use in the URL
        encoded_object_key = quote(object_key)
        download_url = f"{os.getenv('SPACES_ENDPOINT_V2')}/{encoded_object_key}"

        # Generate pre-signed S3 upload URL
        upload_url = generate_presigned_url(
            object_key=object_key,
            expiration=3600,  # URL expiration time in seconds
            http_method="PUT",
            size=size,
            content_type=type,
            public=public,
            bucket=bucket,
            region=region,
            access_key=access_key,
            secret_key=secret_key,
        )

        # if public:
        #     # Set the object ACL to public-read after upload
        #     set_object_acl_public(os.getenv("SPACES_BUCKET_V2"), object_key)

        return {
            "url": upload_url,
            "download_url": download_url,
            "include_acl": public,
            "is_public": public,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
