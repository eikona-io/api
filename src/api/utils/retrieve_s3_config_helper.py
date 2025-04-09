from pydantic import BaseModel
import os
from api.models import UserSettings

global_bucket = os.getenv("SPACES_BUCKET_V2")
global_region = os.getenv("SPACES_REGION_V2")
global_access_key = os.getenv("SPACES_KEY_V2")
global_secret_key = os.getenv("SPACES_SECRET_V2")


class S3Config(BaseModel):
    public: bool
    bucket: str
    region: str
    access_key: str
    secret_key: str


def retrieve_s3_config(user_settings: UserSettings) -> S3Config:
    public = True
    bucket = global_bucket
    region = global_region
    access_key = global_access_key
    secret_key = global_secret_key

    if user_settings is not None:
        if user_settings.output_visibility == "private":
            public = False

        if user_settings.custom_output_bucket:
            bucket = user_settings.s3_bucket_name
            region = user_settings.s3_region
            access_key = user_settings.s3_access_key_id
            secret_key = user_settings.s3_secret_access_key

    return S3Config(
        public=public,
        bucket=bucket,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
    )
