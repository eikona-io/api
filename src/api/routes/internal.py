import json
import os
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

from sqlalchemy import select, update

from api.database import AsyncSessionLocal, get_clickhouse_client, get_db
from api.models import WorkflowRun, WorkflowRunOutput

from fastapi import Query

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


class MyRequest(BaseModel):
    run_id: str
    status: Optional[WorkflowRunStatus] = None
    time: Optional[datetime] = Field(default_factory=datetime.utcnow)
    output_data: Optional[Any] = None
    node_meta: Optional[Any] = None
    log_data: Optional[Any] = None
    logs: Optional[Any] = None
    live_status: Optional[str] = None
    progress: Optional[float] = None
    modal_function_call_id: Optional[str] = None


router = APIRouter()


@router.post("/update-run", include_in_schema=False)
async def update_run(
    request: MyRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSessionLocal = Depends(get_db),
    client: AsyncClient = Depends(get_clickhouse_client),
):
    updated_at = datetime.utcnow()

    # Ensure request.time is timezone-aware
    # fixed_time = request.time.replace(tzinfo=timezone.utc) if request.time and request.time.tzinfo is None else request.time
    fixed_time = updated_at
    
    if request.logs is not None:
        existing_run = await db.execute(
            select(WorkflowRun).where(WorkflowRun.id == request.run_id)
        )
        workflow_run = existing_run.scalar_one_or_none()
        workflow_run = cast(WorkflowRun, workflow_run)

        if not workflow_run:
            raise HTTPException(status_code=404, detail="WorkflowRun not found")
        
        # Sending to clickhouse
        await client.insert(
            table="log_entries",
            data=[
                (
                    uuid4(),
                    request.run_id,
                    workflow_run.workflow_id,
                    workflow_run.machine_id,
                    updated_at,
                    "info",
                    str(request.logs),
                    # request.node_meta.get('node_class', '') if request.node_meta else ''
                )
            ],
        )
        await send_workflow_update(str(request.run_id), {"logs": request.logs})
        return {"status": "success"}

    # Updating the progress
    if request.live_status is not None and request.progress is not None:
        # Updating the workflow run table with live_status, progress, and updated_at
        update_stmt = (
            update(WorkflowRun)
            .where(WorkflowRun.id == request.run_id)
            .values(
                live_status=request.live_status,
                progress=request.progress,
                updated_at=updated_at,
            )
            .returning(WorkflowRun)
        )
        result = await db.execute(update_stmt)
        await db.commit()
        # await db.refresh(WorkflowRun)

        workflow_run = result.scalar_one()
        workflow_run = cast(WorkflowRun, workflow_run)

        # Sending a real-time update to connected clients
        await send_workflow_update(
            str(workflow_run.workflow_id), workflow_run.to_dict()
        )
        await send_realtime_update(str(workflow_run.id), workflow_run.to_dict())

        # Sending to clickhouse
        await client.insert(
            table="progress_updates",
            data=[
                (
                    uuid4(),
                    request.run_id,
                    workflow_run.workflow_id,
                    workflow_run.machine_id,
                    updated_at,
                    request.progress,
                    request.live_status,
                    request.status,
                    # request.node_meta.get('node_class', '') if request.node_meta else ''
                )
            ],
        )

        if workflow_run.webhook and workflow_run.webhook_intermediate_status:
            background_tasks.add_task(
                send_webhook, workflow_run, updated_at, workflow_run.id
            )

        return {"status": "success"}

    if request.log_data is not None:
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

    if request.status == "started" and fixed_time is not None:
        update_stmt = (
            update(WorkflowRun)
            .where(WorkflowRun.id == request.run_id)
            .values(started_at=fixed_time, updated_at=updated_at)
        )
        await db.execute(update_stmt)
        await db.commit()
        # await db.refresh(workflow_run)

    if request.status == "queued" and fixed_time is not None:
        # Ensure request.time is UTC-aware
        update_stmt = (
            update(WorkflowRun)
            .where(WorkflowRun.id == request.run_id)
            .values(queued_at=fixed_time, updated_at=updated_at)
        )
        await db.execute(update_stmt)
        await db.commit()
        # await db.refresh(workflow_run)
        # return {"status": "success"}

    ended = request.status in ["success", "failed", "timeout", "cancelled"]
    if request.output_data is not None:
        # Sending to postgres
        newOutput = WorkflowRunOutput(
            id=uuid4(),  # Add this line to generate a new UUID for the primary key
            created_at=updated_at,
            updated_at=updated_at,
            run_id=request.run_id,
            data=request.output_data,
            node_meta=request.node_meta,
        )
        db.add(newOutput)
        await db.commit()
        await db.refresh(newOutput)
    elif request.status is not None:
        # Get existing run
        existing_run = await db.execute(
            select(WorkflowRun).where(WorkflowRun.id == request.run_id)
        )
        workflow_run = existing_run.scalar_one_or_none()
        workflow_run = cast(WorkflowRun, workflow_run)

        if not workflow_run:
            raise HTTPException(status_code=404, detail="WorkflowRun not found")

        # If the run is already cancelled, don't update it
        if request.status == "cancelled":
            return {"status": "success"}

        # Update the run status
        update_data = {"status": request.status, "updated_at": updated_at}
        if ended and fixed_time is not None:
            update_data["ended_at"] = fixed_time
            
        # Sending to clickhouse
        await client.insert(
            table="progress_updates",
            data=[
                (
                    uuid4(),
                    request.run_id,
                    workflow_run.workflow_id,
                    workflow_run.machine_id,
                    updated_at,
                    request.progress,
                    request.live_status,
                    request.status,
                    # request.node_meta.get('node_class', '') if request.node_meta else ''
                )
            ],
        )

        update_stmt = (
            update(WorkflowRun)
            .where(WorkflowRun.id == request.run_id)
            .values(
                status=request.status,
                ended_at=updated_at if ended else None,
                updated_at=updated_at,
            )
            .returning(WorkflowRun)
        )
        result = await db.execute(update_stmt)
        await db.commit()
        await db.refresh(workflow_run)
        workflow_run = result.scalar_one_or_none()
        workflow_run = cast(WorkflowRun, workflow_run)

        # Get all outputs for the workflow run
        outputs_query = select(WorkflowRunOutput).where(
            WorkflowRunOutput.run_id == request.run_id
        )
        outputs_result = await db.execute(outputs_query)
        outputs = outputs_result.scalars().all()

        # Instead of setting outputs directly, create a new dictionary with all the data
        workflow_run_data = workflow_run.to_dict()
        workflow_run_data["outputs"] = [output.to_dict() for output in outputs]
        # workflow_run_data["status"] = request.status.value

        # Use the dictionary instead of the ORM object
        await send_workflow_update(str(workflow_run.workflow_id), workflow_run_data)
        await send_realtime_update(str(workflow_run.id), workflow_run_data)

        if workflow_run.webhook:
            background_tasks.add_task(
                send_webhook, workflow_run, updated_at, workflow_run.id
            )

        return {"status": "success"}

    return {"status": "success"}


@router.post("/gpu_event")
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
            return Response(
                content=content,
                status_code=response.status,
                headers=dict(response.headers),
            )
            
@router.post("/machine-built")
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


@router.get("/file-upload", include_in_schema=False)
async def get_file_upload_url(
    file_name: str = Query(..., description="Name of the file to upload"),
    run_id: UUID4 = Query(..., description="UUID of the run"),
    size: Optional[int] = Query(None, description="Size of the file in bytes"),
    type: str = Query(..., description="Type of the file"),
    public: bool = Query(
        True, description="Whether to make the file publicly accessible"
    ),
):
    try:
        # Generate the object key
        object_key = f"outputs/runs/{run_id}/{file_name}"

        # Generate pre-signed S3 upload URL
        upload_url = generate_presigned_url(
            bucket=os.getenv("SPACES_BUCKET_V2"),
            object_key=object_key,
            expiration=3600,  # URL expiration time in seconds
            http_method="PUT",
            size=size,
            content_type=type,
            public=public,
        )

        # Generate static download URL
        download_url = f"{os.getenv('SPACES_ENDPOINT_V2')}/{object_key}"

        # if public:
        #     # Set the object ACL to public-read after upload
        #     set_object_acl_public(os.getenv("SPACES_BUCKET_V2"), object_key)

        return {"url": upload_url, "download_url": download_url, "include_acl": public}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_presigned_url(
    bucket,
    object_key,
    expiration=3600,
    http_method="PUT",
    size=None,
    content_type=None,
    public=False,
):
    s3_client = boto3.client(
        "s3",
        region_name=os.getenv("SPACES_REGION_V2"),
        aws_access_key_id=os.getenv("SPACES_KEY_V2"),
        aws_secret_access_key=os.getenv("SPACES_SECRET_V2"),
        config=Config(signature_version="s3v4"),
    )
    params = {
        "Bucket": bucket,
        "Key": object_key,
    }

    if public:
        params["ACL"] = "public-read"

    # if size is not None:
    #     params["ContentLength"] = size
    if content_type is not None:
        params["ContentType"] = content_type

    try:
        response = s3_client.generate_presigned_url(
            ClientMethod=f"{http_method.lower()}_object",
            Params=params,
            ExpiresIn=expiration,
            HttpMethod=http_method,
        )
    except ClientError as e:
        print(e)
        return None

    return response


async def send_workflow_update(workflow_id: str, data: dict):
    logging.info(f"Sending updateWorkflow event via POST: {workflow_id}")
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{os.getenv('NEXT_PUBLIC_REALTIME_SERVER_2')}/updateWorkflow"
            json_data = json.dumps({"workflowId": workflow_id, "data": data})
            async with session.post(
                url, data=json_data, headers={"Content-Type": "application/json"}
            ) as response:
                if response.status >= 400:
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=f"Failed to send update: {response.reason}",
                        headers=response.headers,
                    )
    except Exception as error:
        print(data)
        logging.error(f"Error sending updateWorkflow event: {error}")


async def send_realtime_update(id: str, data: dict):
    logging.info(f"Sending updateWorkflow event via POST: {id}")
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{os.getenv('NEXT_PUBLIC_REALTIME_SERVER_2')}/update"
            json_data = json.dumps({"id": id, "data": data})
            async with session.post(
                url, data=json_data, headers={"Content-Type": "application/json"}
            ) as response:
                if response.status >= 400:
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=f"Failed to send update: {response.reason}",
                        headers=response.headers,
                    )
    except Exception as error:
        print(data)
        logging.error(f"Error sending updateWorkflow event: {error}")


async def fetch_with_timeout(url, options, timeout=20):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                options.get("method", "GET"), url, **options, timeout=timeout
            ) as response:
                return response
    except asyncio.TimeoutError:
        raise TimeoutError("Request timed out")


async def retry_fetch(url, options, num_retries=3):
    for i in range(num_retries):
        try:
            response = await fetch_with_timeout(url, options)
            if not response.ok and i < num_retries - 1:
                continue  # Retry if the response is not ok and retries are left
            return response
        except Exception as error:
            if i == num_retries - 1:
                raise error  # Throw error if it's the last retry
