from typing import Optional
from pydantic import BaseModel
import os
from api.models import UserSettings
import base64
import google.auth
from google.auth.transport import requests
import boto3
import time
import google.auth.transport.requests
import google.oauth2.id_token
from dateutil import parser
import aioboto3
import asyncio
import aiohttp
import logfire

global_bucket = os.getenv("SPACES_BUCKET_V2")
global_region = os.getenv("SPACES_REGION_V2")
global_access_key = os.getenv("SPACES_KEY_V2")
global_secret_key = os.getenv("SPACES_SECRET_V2")

async def get_assumed_role_credentials(assumed_role_arn: str, region: str):
    # Directly fetch ID token from metadata service
    audience = "sts.amazonaws.com"
    metadata_url = f"http://metadata/computeMetadata/v1/instance/service-accounts/default/identity?audience={audience}&format=full"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(metadata_url, headers={"Metadata-Flavor": "Google"}) as response:
            if response.status != 200:
                raise Exception(f"Failed to get ID token: {response.status} {await response.text()}")

            id_token = await response.text()
            logfire.info("ID token", extra={"id_token": id_token})
            
    # Use aioboto3 for async AWS operations
    async with aioboto3.Session().client('sts', region_name=region) as sts_client:
        response = await sts_client.assume_role_with_web_identity(
            RoleArn=assumed_role_arn,
            RoleSessionName='comfydeploy-session',
            WebIdentityToken=id_token
        )
        
        credentials = response['Credentials']
        # Use datetime instead of time.strptime to avoid timezone issues
        expiration_time = parser.isoparse(credentials['Expiration']).timestamp()
        credentials = {
            "access_key": credentials['AccessKeyId'],
            "secret_key": credentials['SecretAccessKey'],
            "session_token": credentials['SessionToken'],
            "expiration": expiration_time
        }
        return credentials


class S3Config(BaseModel):
    public: bool
    bucket: str
    region: str
    access_key: str
    secret_key: str
    is_custom: bool
    session_token: Optional[str] = None


async def retrieve_s3_config(user_settings: UserSettings) -> S3Config:
    public = True
    bucket = global_bucket
    region = global_region
    access_key = global_access_key
    secret_key = global_secret_key
    is_custom = False
    session_token = None
    
    if user_settings is not None:
        if user_settings.output_visibility == "private":
            public = False

        if user_settings.custom_output_bucket:
            bucket = user_settings.s3_bucket_name
            region = user_settings.s3_region
            access_key = user_settings.s3_access_key_id
            secret_key = user_settings.s3_secret_access_key
            is_custom = True
            
        if user_settings.assumed_role_arn:
            credentials = await get_assumed_role_credentials(user_settings.assumed_role_arn, region)
            
            access_key = credentials['access_key']
            secret_key = credentials['secret_key']
            session_token = credentials['session_token']
            # expiration = credentials['expiration']

    return S3Config(
        public=public,
        bucket=bucket,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        is_custom=is_custom,
        session_token=session_token,
    )
