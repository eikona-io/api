from datetime import datetime, timedelta, timezone
from functools import wraps
from http.client import HTTPException
import logging
from typing import Any, Literal, Self, TypeVar, Tuple, Union
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
from urllib.parse import urlparse, unquote
import asyncio
from fastapi import Depends
from fastapi.responses import JSONResponse
import httpx
from functools import lru_cache, wraps
import os
from google.cloud import pubsub_v1
from google.api_core import exceptions
from jose import JWTError, jwt
from datetime import datetime, timedelta
from .subscription import get_current_plan, get_usage_detail
from decimal import Decimal
import math
from typing import Dict, List, Set

# Get JWT secret from environment variable
JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"


def generate_temporary_token(
    user_id: str, org_id: Optional[str] = None, expires_in: str = "1h"
) -> str:
    """
    Generate a temporary JWT token for the given user_id and org_id.

    Args:
        user_id (str): The user ID to include in the token.
        org_id (Optional[str]): The organization ID to include in the token, if any.
        expires_in (str): The expiration time for the token. Default is "1h".

    Returns:
        str: The generated JWT token.
    """
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=1),  # Default expiration of 1 hour
    }

    if org_id:
        payload["org_id"] = org_id

    if expires_in != "1h":
        # Parse the expiration time
        value = int(expires_in[:-1])
        unit = expires_in[-1].lower()
        if unit == "m":
            delta = timedelta(minutes=value)
        elif unit == "h":
            delta = timedelta(hours=value)
        elif unit == "d":
            delta = timedelta(days=value)
        elif unit == "w":
            delta = timedelta(weeks=value)
        else:
            raise ValueError("Invalid expiration format. Use m, h, d, or w.")

        payload["exp"] = datetime.utcnow() + delta

    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


Base = declarative_base()

T = TypeVar("T")

clerk_token = os.getenv("CLERK_SECRET_KEY")


def async_lru_cache(maxsize=128, typed=False, expire_after=None):
    def decorator(async_func):
        cache = {}

        @wraps(async_func)
        async def wrapper(*args, **kwargs):
            cache_key = args + tuple(sorted(kwargs.items()))
            now = datetime.now()

            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if expire_after is None or now - timestamp < expire_after:
                    return result

            result = await async_func(*args, **kwargs)
            cache[cache_key] = (result, now)

            if len(cache) > maxsize:
                oldest_key = min(cache, key=lambda k: cache[k][1])
                del cache[oldest_key]

            return result

        return wrapper

    return decorator


# Add this cache at the module level
user_icon_cache = {}


@async_lru_cache(maxsize=1000)
async def fetch_user_icon(user_id: str) -> tuple[str, Optional[str]]:
    current_time = datetime.now()

    # Check if the user_id is in the cache and not expired (1 day)
    if user_id in user_icon_cache:
        cached_data, timestamp = user_icon_cache[user_id]
        if current_time - timestamp < timedelta(days=1):
            return user_id, cached_data

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.clerk.com/v1/users/{user_id}",
            headers={"Authorization": f"Bearer {clerk_token}"},
        )
        if response.status_code == 200:
            user_data = response.json()
            image_url = user_data.get("image_url")
            # Update the cache
            user_icon_cache[user_id] = (image_url, current_time)
            return user_id, image_url

    # If fetching fails, cache None for 1 hour to avoid frequent retries
    user_icon_cache[user_id] = (None, current_time)
    return user_id, None


def get_org_or_user_condition(target: Base, request: Request):
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

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

    def apply_org_check_by_type(self, type, request: Request) -> Self:
        return self.where(get_org_or_user_condition(type, request))

    def paginate(self, limit: int, offset: int) -> Self:
        return self.limit(limit).offset(offset)


def select(__ent0: _ColumnsClauseArgument[T], /, *entities: Any) -> OrgAwareSelect[T]:
    return OrgAwareSelect(__ent0, *entities)


def ensure_run_timeout(run):
    # Apply timeout logic
    timeout_hours = 24
    timeout_delta = timedelta(hours=timeout_hours)
    now = datetime.now(timezone.utc)

    # Not started for 24 hours
    if (
        run.status == "not-started"
        and now - run.created_at.replace(tzinfo=timezone.utc) > timeout_delta
    ):
        run.status = "timeout"

    # Queued for 24 hours
    elif (
        run.status == "queued"
        and run.queued_at
        and now - run.queued_at.replace(tzinfo=timezone.utc) > timeout_delta
    ):
        run.status = "timeout"

    # Started for 24 hours
    elif (
        run.status == "started"
        and run.started_at
        and now - run.started_at.replace(tzinfo=timezone.utc) > timeout_delta
    ):
        run.status = "timeout"

    # Running and not updated in the last 24 hours
    elif run.status not in ["success", "failed", "timeout", "cancelled"]:
        updated_at = (
            run.updated_at.replace(tzinfo=timezone.utc)
            if run.updated_at.tzinfo is None
            else run.updated_at
        )
        if now - updated_at > timeout_delta:
            run.status = "timeout"


def post_process_outputs(outputs, user_settings):
    for output in outputs:
        if output.data and isinstance(output.data, dict):
            post_process_output_data(output.data, user_settings)


def post_process_output_data(data, user_settings):
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

    for upload_type in ["images", "files", "gifs", "mesh"]:
        if upload_type in data:
            for output_item in data[upload_type]:
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
    return data


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
    object_key = unquote(parsed_url.path.lstrip("/"))

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


project_id = os.getenv("GOOGLE_CLOUD_PROJECT")


async def send_workflow_update(workflow_id: str, data: dict):
    return
    logging.info(f"Sending updateWorkflow event via POST: {workflow_id}")
    # try:
    #     topic_id = "workflow-updates"

    #     # Create the topic if it doesn't exist
    #     # await create_topic_if_not_exists(project_id, topic_id)

    #     publisher = pubsub_v1.PublisherClient()
    #     # Create a publisher client
    #     topic_path = publisher.topic_path(project_id, topic_id)

    #     # Prepare the message
    #     message_data = json.dumps({"workflowId": workflow_id, "data": data}).encode(
    #         "utf-8"
    #     )

    #     # Publish the message
    #     publish_future = publisher.publish(
    #         topic_path,
    #         data=message_data,
    #         id=str(workflow_id),  # Add the ID as a message attribute
    #     )
    #     # Use asyncio to wait for the future without blocking
    #     message_id = await asyncio.wrap_future(publish_future)

    #     logging.info(f"Published message with ID: {message_id}")
    # except Exception as error:
    #     print(data)
    #     logging.error(f"Error sending updateWorkflow event: {error}")

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


async def create_topic_if_not_exists(project_id: str, topic_id: str):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    try:
        publisher.create_topic(request={"name": topic_path})
        print(f"Topic {topic_path} created.")
    except exceptions.AlreadyExists:
        print(f"Topic {topic_path} already exists.")
    except Exception as e:
        print(f"Error creating topic: {e}")
        raise


async def send_realtime_update(id: str, data: dict):
    return
    logging.info(f"Sending updateWorkflow event via POST: {id}")

    # try:
    #     project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    #     topic_id = "realtime-updates"  # Single topic for all realtime updates

    #     # Create the topic if it doesn't exist
    #     # await create_topic_if_not_exists(project_id, topic_id)
    #     publisher = pubsub_v1.PublisherClient()

    #     # Create a publisher client
    #     topic_path = publisher.topic_path(project_id, topic_id)

    #     # Prepare the message
    #     message_data = json.dumps({"id": id, "data": data}).encode("utf-8")

    #     # Publish the message
    #     # Publish the message asynchronously with the ID as an attribute
    #     publish_future = publisher.publish(
    #         topic_path,
    #         data=message_data,
    #         id=str(id),  # Add the ID as a message attribute
    #     )
    #     # Use asyncio to wait for the future without blocking
    #     message_id = await asyncio.wrap_future(publish_future)

    #     logging.info(f"Published realtime update message with ID: {message_id}")
    # except Exception as error:
    #     print(data)
    #     logging.error(f"Error sending realtime update event: {error}")

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
            # logging.info(options)
            method = options.pop('method', 'GET')  # Extract method from options
            async with session.request(
                method, url, **options, timeout=timeout
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


PermissionType = Literal[
    # API Permissions
    "api:runs:get",
    "api:runs:create",
    #
    "api:runs:update",
    "api:file_upload:get",
    #
    "api:machines:update",
    #
    "api:gpu_event:create",
    "api:gpu_event:update",
]


def require_permission(permission: Union[PermissionType, list[PermissionType]]):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            if not has(request, permission):
                # raise HTTPException(status_code=403, detail="Permission denied")
                return JSONResponse(
                    status_code=403, content={"detail": "Permission denied"}
                )
            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


def has(
    request: Request, permission: Union[PermissionType, list[PermissionType]]
) -> bool:
    current_user = request.state.current_user

    if current_user is None:
        return False

    user_permissions = current_user.get("org_permissions", [])

    if isinstance(permission, str):
        return permission in user_permissions
    elif isinstance(permission, list):
        return all(perm in user_permissions for perm in permission)


def r(price: float) -> Decimal:
    return Decimal(math.ceil(price * 1000000) / 1000000)


pricing_lookup_table = {
    "T4": r(0.000164 * 1.10),
    "L4": r(0.000291 * 1.10),
    "A10G": r(0.000306 * 1.10),
    "A100": r(0.001036 * 1.10),
    "A100-80GB": r(0.001553 * 1.10),
    "H100": r(0.002125 * 1.10),
    "CPU": r(0.000038 * 1.10),
}


async def get_total_gpu_compute(usage_details: List[Dict[str, Any]]) -> Decimal:
    total_compute_cost = Decimal("0")
    individual_costs: List[Dict[str, Any]] = []

    for usage_detail in usage_details:
        gpu = usage_detail.gpu
        if not gpu or gpu not in pricing_lookup_table:
            print(f"GPU type {gpu} not found in pricing lookup table")
            continue

        usage_in_sec = usage_detail.usage_in_sec or Decimal("0")
        if usage_in_sec < 0:
            print("Usage time cannot be negative")
            continue

        if not usage_in_sec or usage_in_sec.is_nan():
            print("Usage time is NaN")
            continue

        machine_name = usage_detail.machine_name
        machine_cost = pricing_lookup_table[gpu] * usage_in_sec
        total_compute_cost += machine_cost

        individual_costs.append({
            "machine_name": machine_name,
            "gpu": gpu,
            "usage_in_sec": usage_in_sec,
            "cost": machine_cost
        })

    # Print debug information
    # for cost_detail in individual_costs:
    #     print(f"Machine: {cost_detail['machine_name']}, GPU: {cost_detail['gpu']}, "
    #           f"Usage: {cost_detail['usage_in_sec']:.2f} sec, Cost: ${cost_detail['cost']:.6f}")

    # print(f"Total Compute Cost: ${total_compute_cost:.6f}")

    return total_compute_cost


async def is_exceed_spend_limit(request: Request, db: AsyncSession):
    # default spend limit
    spend_limit = 500
    # max_spend_limit = 1000

    user_settings = await get_user_settings(request, db)

    if user_settings is None:
        return False

    current_plan = await get_current_plan(request, db, select)
    last_invoice_timestamp = current_plan.last_invoice_timestamp

    usage_details = await get_usage_detail(request, last_invoice_timestamp, db, select)

    # print("usage_details:")
    # for item in usage_details:
    #     print(item)
    # print()

    total_compute_cost = await get_total_gpu_compute(usage_details)
    print("total_compute_cost", total_compute_cost)
    
    user_spend_limit = Decimal(user_settings.spend_limit if user_settings.spend_limit is not None else spend_limit)
    
    if total_compute_cost > user_spend_limit:
        return True
    
    return False
