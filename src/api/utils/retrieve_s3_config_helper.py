from typing import Optional
from pydantic import BaseModel
import os
from api.models import UserSettings
import base64
import google.auth
from google.auth.transport import requests
import google.oauth2.id_token
import boto3
import time

global_bucket = os.getenv("SPACES_BUCKET_V2")
global_region = os.getenv("SPACES_REGION_V2")
global_access_key = os.getenv("SPACES_KEY_V2")
global_secret_key = os.getenv("SPACES_SECRET_V2")

async def get_assumed_role_credentials(assumed_role_arn: str):
    # This should apply cache if the expiration is not expired
    
    credentials, project = google.auth.default()
    request = requests.Request()
    credentials.refresh(request)
    
    # Get an ID token from the credentials
    id_token_creds = google.oauth2.id_token.fetch_id_token(request, "https://sts.amazonaws.com")
    
    sts_client = boto3.client('sts')
    response = sts_client.assume_role_with_web_identity(
        RoleArn=assumed_role_arn,
        RoleSessionName='comfydeploy-session',
        WebIdentityToken=id_token_creds
    ) 
    credentials = response['Credentials']
    expiration_time = time.mktime(
        time.strptime(credentials['Expiration'], "%Y-%m-%dT%H:%M:%S%Z")
    )
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
            credentials = await get_assumed_role_credentials(user_settings.assumed_role_arn)
            
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
