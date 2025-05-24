from fastapi import APIRouter, Query, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
import re
from typing import Dict, Optional, Any
import hashlib
import asyncio
import logfire

from api.models import UserSettings
from api.utils.retrieve_s3_config_helper import retrieve_s3_config, S3Config
# from src.modal_apps.image_optimizer import optimize_image
from .utils import (
    get_user_settings, 
    generate_presigned_url,
    generate_presigned_download_url
)
from api.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
import modal

router = APIRouter()

@router.get("/optimize/{transformations}/{s3_key:path}")
async def optimize_image_on_demand(
    transformations: str,
    s3_key: str,
    request: Request,
    cd_token: str = Query(None),
    # cache: int = Query(86400, description="Cache duration in seconds"),
    db: AsyncSession = Depends(get_db)
):
    cache = 86400
    """
    On-demand image optimization with Cloudflare-like URL structure
    
    Examples:
    - /optimize/w_800,h_600,q_80,f_webp/uploads/user123/image.jpg
    - /optimize/auto/profile-pics/avatar.jpg
    
    Query parameters:
    - cd_token: Authentication token (if required)
    - cache: Cache duration in seconds (default: 24 hours)
    """
    
    try:
        # Parse transformation parameters
        transform_config = parse_transformations(transformations)
        
        # Get user settings and S3 configuration
        user_settings = await get_user_settings(request, db)
        s3_config = await retrieve_s3_config(user_settings)
        
        # Generate cache key for optimized image
        cache_key = generate_cache_key(s3_key, transform_config)
        
        # Extract the file extension from the original key
        file_extension = get_file_extension(s3_key)
        
        # If format is specified in transformations, use that extension instead
        if "format" in transform_config:
            file_extension = f".{transform_config['format']}"
            
        optimized_key = f"optimized/{cache_key}{file_extension}"
        
        # Check if optimized version exists
        if await check_s3_object_exists(s3_config, optimized_key):
            logfire.info("Serving existing optimized image", extra={
                "s3_key": s3_key,
                "optimized_key": optimized_key,
                "transformations": transformations
            })
            return await get_optimized_image_response(s3_config, optimized_key, user_settings, cache)
        
        # Check if original image exists
        if not await check_s3_object_exists(s3_config, s3_key):
            raise HTTPException(status_code=404, detail="Original image not found")
        
        # Trigger optimization asynchronously
        await trigger_image_optimization(s3_config, s3_key, optimized_key, transform_config)
        
        logfire.info("Triggered image optimization", extra={
            "s3_key": s3_key,
            "optimized_key": optimized_key,
            "transformations": transformations
        })
        
        # Return URL to optimized image (will be ready shortly)
        return await get_optimized_image_response(s3_config, optimized_key, user_settings, cache)
        
    except HTTPException:
        raise
    except Exception as e:
        logfire.error("Image optimization request failed", extra={
            "s3_key": s3_key,
            "transformations": transformations,
            "error": str(e)
        })
        # Fallback to original image if we have s3_config
        user_settings = await get_user_settings(request, db)
        s3_config = await retrieve_s3_config(user_settings)
        return await get_fallback_response(s3_config, s3_key, user_settings, cache)


def parse_transformations(transformations: str) -> Dict[str, Any]:
    """Parse transformation string into config dict"""
    if transformations == "auto":
        return {
            "format": "webp",
            "quality": 85,
            "max_width": 1920,
            "max_height": 1080,
            "auto_optimize": True
        }
    
    config = {}
    params = transformations.split(",")
    
    for param in params:
        if param.startswith("w_"):
            config["max_width"] = int(param[2:])
        elif param.startswith("h_"):
            config["max_height"] = int(param[2:])
        elif param.startswith("q_"):
            config["quality"] = int(param[2:])
        elif param.startswith("f_"):
            config["format"] = param[2:]
    
    return config


def generate_cache_key(s3_key: str, config: Dict[str, Any]) -> str:
    """Generate deterministic cache key for optimized image"""
    # Sort config for consistent hashing
    config_items = sorted(config.items())
    config_str = "_".join(f"{k}-{v}" for k, v in config_items)
    content = f"{s3_key}_{config_str}"
    
    # Use first 16 chars of MD5 hash for shorter keys
    hash_obj = hashlib.md5(content.encode())
    return hash_obj.hexdigest()[:16]


def get_file_extension(s3_key: str) -> str:
    """Extract file extension from S3 key"""
    # Extract extension (e.g., .jpg, .png) including the dot
    match = re.search(r'\.[^.]+$', s3_key)
    return match.group(0) if match else ""


async def trigger_image_optimization(
    s3_config: S3Config,
    original_key: str, 
    optimized_key: str, 
    transform_config: Dict[str, Any]
):
    """Trigger Modal optimization in background"""
    
    try:
        # Generate presigned URLs for Modal (5 minutes expiry)
        input_url = generate_presigned_download_url(
            bucket=s3_config.bucket,
            object_key=original_key,
            region=s3_config.region,
            access_key=s3_config.access_key,
            secret_key=s3_config.secret_key,
            session_token=s3_config.session_token,
            expiration=300
        )
        
        output_url = generate_presigned_url(
            bucket=s3_config.bucket,
            object_key=optimized_key,
            region=s3_config.region,
            access_key=s3_config.access_key,
            secret_key=s3_config.secret_key,
            session_token=s3_config.session_token,
            expiration=300,
            http_method="PUT"
        )
        
        optimize_image = modal.Function.from_name("image-optimizer", "optimize_image")
        
        # Call Modal function asynchronously (fire and forget)
        await optimize_image.remote.aio(input_url, output_url, transform_config)
        
    except Exception as e:
        logfire.error("Failed to trigger image optimization", extra={
            "original_key": original_key,
            "optimized_key": optimized_key,
            "error": str(e)
        })
        raise


async def get_optimized_image_response(
    s3_config: S3Config, 
    optimized_key: str, 
    user_settings: Optional[UserSettings],
    cache_duration: int = 86400  # Default cache duration of 24 hours (in seconds)
):
    """Return appropriate response for optimized image"""
    
    # Create response headers with cache control
    headers = {
        "Cache-Control": f"public, max-age={cache_duration}, stale-while-revalidate=60",
        "Vary": "Accept-Encoding"
    }
    
    if s3_config.public:
        # Public bucket - return direct URL
        public_url = f"https://{s3_config.bucket}.s3.{s3_config.region}.amazonaws.com/{optimized_key}"
        return RedirectResponse(url=public_url, status_code=302, headers=headers)
    else:
        # Private bucket - return presigned URL
        presigned_url = generate_presigned_download_url(
            bucket=s3_config.bucket,
            object_key=optimized_key,
            region=s3_config.region,
            access_key=s3_config.access_key,
            secret_key=s3_config.secret_key,
            session_token=s3_config.session_token,
            expiration=3600  # 1 hour
        )
        return RedirectResponse(url=presigned_url, status_code=302, headers=headers)


async def get_fallback_response(
    s3_config: S3Config,
    s3_key: str, 
    user_settings: Optional[UserSettings],
    cache_duration: int = 43200  # Default 12 hours for fallback images
):
    """Fallback to serving original image if optimization fails"""
    logfire.info("Serving original image as fallback", extra={"s3_key": s3_key})
    return await get_optimized_image_response(s3_config, s3_key, user_settings, cache_duration)


async def check_s3_object_exists(s3_config: S3Config, s3_key: str) -> bool:
    """Check if S3 object exists"""
    import aioboto3
    from botocore.exceptions import ClientError
    
    try:
        session = aioboto3.Session()
        async with session.client(
            's3',
            region_name=s3_config.region,
            aws_access_key_id=s3_config.access_key,
            aws_secret_access_key=s3_config.secret_key,
            aws_session_token=s3_config.session_token
        ) as s3:
            await s3.head_object(Bucket=s3_config.bucket, Key=s3_key)
            return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise