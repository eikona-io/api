from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Self, TypeVar, Tuple
from fastapi import Request
from sqlalchemy import GenerativeSelect, Select
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql.selectable import _ColumnsClauseArgument
import os
from typing import Optional, cast
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.models import UserSettings
from pprint import pprint
import json
import aiohttp
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from urllib.parse import urlparse
import asyncio
from fastapi import Depends

Base = declarative_base()

T = TypeVar("T")


def get_org_or_user_condition(target: Base, request: Request):
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"]

    return (
        (target.org_id == org_id)
        if org_id
        else ((target.user_id == user_id) & (target.org_id.is_(None)))
    )


class OrgAwareSelect(Select[Tuple[T]]):
    inherit_cache = True

    def apply_org_check(self, request: Request) -> Self:
        return self.where(
            get_org_or_user_condition(self.column_descriptions[0]["entity"], request)
        )

    def paginate(self, limit: int, offset: int) -> Self:
        return self.limit(limit).offset(offset)


def select(__ent0: _ColumnsClauseArgument[T], /, *entities: Any) -> OrgAwareSelect[T]:
    return OrgAwareSelect(__ent0, *entities)


def ensure_run_timeout(run):
    # Apply timeout logic
    timeout_minutes = 15
    timeout_delta = timedelta(minutes=timeout_minutes)
    now = datetime.now(timezone.utc)

    # Not started for 15 mins
    if (
        run.status == "not-started"
        and now - run.created_at.replace(tzinfo=timezone.utc) > timeout_delta
    ):
        run.status = "timeout"

    # Queued for 15 mins
    elif (
        run.status == "queued"
        and run.queued_at
        and now - run.queued_at.replace(tzinfo=timezone.utc) > timeout_delta
    ):
        run.status = "timeout"

    # Started for 15 mins
    elif (
        run.status == "started"
        and run.started_at
        and now - run.started_at.replace(tzinfo=timezone.utc) > timeout_delta
    ):
        run.status = "timeout"

    # Running and not updated in the last 15 mins
    elif run.status not in ["success", "failed", "timeout", "cancelled"]:
        updated_at = (
            run.updated_at.replace(tzinfo=timezone.utc)
            if run.updated_at.tzinfo is None
            else run.updated_at
        )
        if now - updated_at > timeout_delta:
            run.status = "timeout"


def post_process_outputs(outputs, user_settings):
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

    for output in outputs:
        if output.data and isinstance(output.data, dict):
            # Check if 'is_public' is False in the output data
            # pprint(output.data)
            for upload_type in ["images", "files", "gifs", "mesh"]:
                if upload_type in output.data:
                    for output_item in output.data[upload_type]:
                        if output_item.get("is_public") is False:
                            # pprint(output_item)
                            # Replace the output URL with a session key
                            if "url" in output_item:
                                output_item["url"] = get_temporary_download_url(
                                    output_item["url"],
                                    region,
                                    access_key,
                                    secret_key,
                                )
                                # print(output_item["url"])


async def get_user_settings(request: Request, db: AsyncSession):
    user_query = select(UserSettings).apply_org_check(request)
    user_settings = await db.execute(user_query)
    user_settings = user_settings.scalar_one_or_none()
    user_settings = cast(Optional[UserSettings], user_settings)
    return user_settings


def get_temporary_download_url(
    url: str, region: str, access_key: str, secret_key: str, expiration: int = 3600
) -> str:
    # Parse the URL
    parsed_url = urlparse(url)

    # Extract bucket name and object key
    bucket = parsed_url.netloc.split(".")[0]
    object_key = parsed_url.path.lstrip("/")

    # Generate and return the presigned URL
    return generate_presigned_download_url(
        bucket=bucket,
        object_key=object_key,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        expiration=expiration,
    )


def generate_presigned_download_url(
    bucket: str,
    object_key: str,
    region: str,
    access_key: str,
    secret_key: str,
    expiration: int = 3600,
):
    s3_client = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )

    try:
        response = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": bucket,
                "Key": object_key,
            },
            ExpiresIn=expiration,
        )
        return response
    except ClientError as e:
        logging.error(f"Error generating presigned download URL: {e}")
        return None


def generate_presigned_url(
    bucket,
    object_key,
    region: str,
    access_key: str,
    secret_key: str,
    expiration=3600,
    http_method="PUT",
    size=None,
    content_type=None,
    public=False,
):
    s3_client = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
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
