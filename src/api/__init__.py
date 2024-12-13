from typing import Optional
from api.middleware.subscriptionMiddleware import SubscriptionMiddleware
from sqlalchemy import select

import modal
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse  # Import JSONResponse from here
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from jose import JWTError, jwt
from sqlalchemy.orm import Session  # Import Session from SQLAlchemy
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import AsyncSessionLocal, init_db, get_db, engine
from api.routes import (
    run,
    volumes,
    internal,
    workflow,
    log,
    workflows,
    machines,
    comfy_node,
    deployments,
    runs,
    session,
    files,
    models,
    platform,
    search,
)
from api.modal import builder
from api.models import APIKey
from dotenv import load_dotenv
import logfire
import logging
from scalar_fastapi import get_scalar_api_reference
from fastapi.openapi.utils import get_openapi
from fastapi import APIRouter

# import all you need from fastapi-pagination
# from fastapi_pagination import Page, add_pagination, paginate
from pprint import pprint
from api.middleware.authMiddleware import AuthMiddleware
from api.middleware.spendLimitMiddleware import SpendLimitMiddleware

from opentelemetry.context import get_current
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.textmap import TextMapPropagator


class NullPropagator(TextMapPropagator):
    def extract(self, *args, **kwargs):
        return get_current()

    def inject(self, *args, **kwargs):
        pass

    @property
    def fields(self):
        return set()


set_global_textmap(NullPropagator())

load_dotenv()
logfire.configure(
    service_name="comfydeploy-api",
    # send_to_logfire=False
)
logger = logfire
logging.basicConfig(level=logging.INFO)

# logger = logging.getLogger(__name__)

# Load environment variables from .env file

# Replace the existing oauth2_scheme declaration with this:
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        # "read:runs": "Read access to runs",
        # "write:runs": "Write access to runs",
        # Add more scopes as needed
    },
)


app = FastAPI(
    servers=[
        {"url": "https://api.comfydeploy.com/api", "description": "Production server"},
        {
            "url": "https://staging.api.comfydeploy.com/api",
            "description": "Staging server",
        },
        {"url": "http://localhost:3011/api", "description": "Local development server"},
    ]
)

# add_pagination(app)

app.openapi_schema = None  # Clear any existing schema

docs = """
### Overview

Welcome to the ComfyDeploy API!

To create a run thru the API, use the [queue run endpoint](#tag/run/POST/run/deployment/queue).

Check out the [get run endpoint](#tag/run/GET/run/{run_id}), for getting the status and output of a run.

### Authentication

To authenticate your requests, include your API key in the `Authorization` header as a bearer token. Make sure to generate an API key in the [API Keys section of your ComfyDeploy account](https://www.comfydeploy.com/api-keys).

###

"""


def custom_openapi(with_code_samples: bool = True):
    if app.openapi_schema and with_code_samples:
        return app.openapi_schema
    
    if (with_code_samples and os.getenv("ENV", "production").lower() == "production"):
        # In development mode, fetch from Speakeasy
        import requests
        try:
            response = requests.get("https://spec.speakeasy.com/comfydeploy/comfydeploy/comfydeploy-api-with-code-samples")
            openapi_schema = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch Speakeasy schema: {e}")
            # Fallback to default schema generation
            openapi_schema = get_openapi(
                title="ComfyDeploy API",
                version="V2",
                description=docs,
                routes=public_api_router.routes,
                servers=app.servers,
                webhooks=app.webhooks.routes,
            )
    else:
        openapi_schema = get_openapi(
            title="ComfyDeploy API",
            version="V2",
            description=docs,
            routes=public_api_router.routes,
            servers=app.servers,
            webhooks=app.webhooks.routes,
        )

        openapi_schema["components"]["securitySchemes"] = {
            "Bearer": {
                "type": "http",
                "scheme": "bearer",
            }
        }
         
        # openapi_schema["x-speakeasy-webhooks"] = {
        #     "security": {
        #         "type": "signature",
        #         "name": "x-signature",
        #         "encoding": "base64",
        #         "algorithm": "hmac-sha256",
        #     }
        # }

    # Apply Bearer Auth security globally
    openapi_schema["security"] = [{"Bearer": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def custom_openapi_internal():
    openapi_schema = get_openapi(
        title="ComfyDeploy API (Internal)",
        version="V2",
        description=docs,
        routes=api_router.routes,
        webhooks=app.webhooks.routes,
        servers=app.servers,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
        }
    }

    # Apply Bearer Auth security globally
    openapi_schema["security"] = [{"Bearer": []}]

    return openapi_schema


app.openapi = custom_openapi

# logfire.install_auto_tracing()


api_router = APIRouter()  # Remove the prefix here
public_api_router = APIRouter()  # Remove the prefix here

app.webhooks.include_router(run.webhook_router)

# Include routers
api_router.include_router(run.router)
api_router.include_router(internal.router)
api_router.include_router(workflows.router)
api_router.include_router(workflow.router)
api_router.include_router(builder.router)
api_router.include_router(machines.router)
api_router.include_router(machines.public_router)
api_router.include_router(log.router)
api_router.include_router(volumes.router)
api_router.include_router(comfy_node.router)
api_router.include_router(deployments.router)
api_router.include_router(session.router)
api_router.include_router(session.beta_router)
api_router.include_router(runs.router)
api_router.include_router(files.router)
api_router.include_router(models.router)
api_router.include_router(platform.router)
api_router.include_router(search.router)

# This is for the docs generation
public_api_router.include_router(run.router)
public_api_router.include_router(session.router)
public_api_router.include_router(machines.public_router)
# public_api_router.include_router(session.beta_router)
public_api_router.include_router(deployments.router)
public_api_router.include_router(files.router)
public_api_router.include_router(models.router)
public_api_router.include_router(search.router)
# public_api_router.include_router(platform.router)
# public_api_router.include_router(run.webhook_router)

app.include_router(api_router, prefix="/api")  # Add the prefix here instead


@app.get("/", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
        scalar_proxy_url="https://proxy.scalar.com",
        hide_models=True,
        servers=[
            {"url": server["url"]} for server in app.servers
        ],  # Remove "/api" here
    )
    
    
@app.get("/openapi.json/with-no-code-samples", include_in_schema=False)
async def openapi_json():
    return JSONResponse(status_code=200, content=custom_openapi(with_code_samples=False))


@app.get("/internal/openapi.json", include_in_schema=False)
async def openapi_json_internal():
    return JSONResponse(status_code=200, content=custom_openapi_internal())


@app.get("/internal", include_in_schema=False)
async def scalar_html_internal():
    return get_scalar_api_reference(
        openapi_url="/internal/openapi.json",
        title=app.title,
        scalar_proxy_url="https://proxy.scalar.com",
        servers=[
            {"url": server["url"]} for server in app.servers
        ],  # Remove "/api" here
    )


# Set up logging

# # Hide sqlalchemy logs
# logging.getLogger('sqlalchemy').setLevel(logging.ERROR)

# Add CORS middleware
app.add_middleware(SpendLimitMiddleware)
app.add_middleware(SubscriptionMiddleware)
app.add_middleware(AuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

logfire.instrument_fastapi(app)
logfire.instrument_sqlalchemy(
    engine=engine.sync_engine,
)

if __name__ == "__main__":
    reload = os.getenv("ENV", "production").lower() == "development"
    port = int(os.getenv("PORT", 8000))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    # print("hiii",project_root)
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        workers=4,
        reload=reload,
        reload_dirs=[project_root + "/api/src"],
    )
