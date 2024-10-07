from enum import Enum
import os
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import (
    get_temporary_download_url,
    get_user_settings,
)

# from sqlalchemy import select
from api.database import get_db
from typing import Optional
import logging
from typing import Optional
from botocore.config import Config
import random
import aioboto3
import mimetypes


# Implement nanoid-like function
def custom_nanoid(size=16):
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return "".join(random.choice(alphabet) for _ in range(size))


# Define prefixes
prefixes = {"img": "img", "zip": "zip", "vid": "vid", "file": "file"}


def new_id(prefix):
    return f"{prefixes[prefix]}_{custom_nanoid(16)}"


logger = logging.getLogger(__name__)

router = APIRouter(tags=["File"])


class UploadType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class FileUploadResponse(BaseModel):
    message: str = Field(
        ..., description="A message indicating the result of the file upload"
    )
    file_id: str = Field(..., description="The unique identifier for the uploaded file")
    file_name: str = Field(..., description="The original name of the uploaded file")
    file_url: str = Field(
        ..., description="The URL where the uploaded file can be accessed"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "File uploaded successfully",
                "file_id": "img_1a2b3c4d5e6f7g8h",
                "file_name": "example_image.jpg",
                "file_url": "https://your-bucket.s3.your-region.amazonaws.com/inputs/img_1a2b3c4d5e6f7g8h.jpg",
            }
        }
    }


class FileUploadRequest(BaseModel):
    file: UploadFile = File(...)
    run_id: Optional[str] = Query(None),
    upload_type: UploadType = Query(UploadType.INPUT),
    file_type: Optional[str] = Query(None, regex="^(image/|video/|application/)"),

# Return the session tunnel url
@router.post(
    "/file/upload",
    openapi_extra={
        "x-speakeasy-name-override": "upload",
    },
)
async def upload_file(
    request: Request,
    db: AsyncSession = Depends(get_db),
    file: UploadFile = File(...),
) -> FileUploadResponse:
    # if not file_type:
    # Infer file type from the filename
    inferred_type, _ = mimetypes.guess_type(file.filename)
    file_type = (
        inferred_type or "application/octet-stream"
    )  # Default to binary data if type can't be guessed

    # Check if the file type is allowed
    if not file_type.startswith(("image/", "video/", "application/")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    user_settings = await get_user_settings(request, db)

    bucket = os.getenv("SPACES_BUCKET_V2")
    region = os.getenv("SPACES_REGION_V2")
    access_key = os.getenv("SPACES_KEY_V2")
    secret_key = os.getenv("SPACES_SECRET_V2")
    public = True

    if user_settings is not None:
        if user_settings.output_visibility == "private":
            public = False

        if user_settings.custom_output_bucket:
            bucket = user_settings.s3_bucket_name
            region = user_settings.s3_region
            access_key = user_settings.s3_access_key_id
            secret_key = user_settings.s3_secret_access_key

    # File size check
    file_size = file.size
    size_limit = 50 * 1024 * 1024  # 50MB
    if file_size > size_limit:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {size_limit // (1024 * 1024)}MB limit",
        )

    # Generate file ID and path
    file_extension = os.path.splitext(file.filename)[1]
    if file_type.startswith("video/"):
        file_id = new_id("vid")
    elif file_type.startswith("image/"):
        file_id = new_id("img")
    elif file_type.startswith("application/zip"):
        file_id = new_id("zip")
    else:
        file_id = new_id("file")

    # if upload_type == UploadType.OUTPUT and run_id:
    #     file_path = f"outputs/runs/{run_id}/{file_id}{file_extension}"
    # else:
    file_path = f"inputs/{file_id}{file_extension}"

    async with aioboto3.Session().client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    ) as s3_client:
        try:
            file_content = await file.read()
            await s3_client.put_object(
                Bucket=bucket,
                Key=file_path,
                Body=file_content,
                ACL="public-read" if public else "private",
                ContentType=file_type,
            )

            file_url = f"https://{bucket}.s3.{region}.amazonaws.com/{file_path}"

            if not public:
                file_url = get_temporary_download_url(
                    file_url,
                    region,
                    access_key,
                    secret_key,
                    expiration=3600,  # Set expiration to 1 hour
                )

            # TODO: Implement PostHog event capture here if needed

            return {
                "message": "File uploaded successfully",
                "file_id": file_id,
                "file_name": file.filename,
                "file_url": file_url,
            }
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise HTTPException(status_code=500, detail="Error uploading file")
