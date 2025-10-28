from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.openapi.utils import get_openapi
from fastapi import APIRouter
from pydantic import BaseModel
from scalar_fastapi import get_scalar_api_reference
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import os
from typing import TYPE_CHECKING
from contextlib import asynccontextmanager
import re

from api.autumn_mount import autumn_app

if TYPE_CHECKING:
    pass


class TrailingSlashMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle trailing slash normalization for Autumn ASGI routes.
    This prevents 307 redirects by ensuring trailing slashes are properly handled.
    """
    async def dispatch(self, request: Request, call_next):
        # Only handle requests to /api/autumn/*
        if request.url.path.startswith("/api/autumn/"):
            path = request.url.path
            
            # Since ALL Autumn ASGI routes start with /api/autumn/ and most need trailing slashes,
            # we'll add trailing slashes to all routes except specific exceptions
            autumn_route_exceptions = [
                # Add any routes that specifically DON'T need trailing slashes
                r"^/api/autumn/customers/[^/]+/entities/[^/]+$",  # DELETE route doesn't use trailing slash
                r"^/api/autumn/webhook$",  # DELETE route doesn't use trailing slash
            ]
            
            # Check if this path needs a trailing slash
            if not path.endswith("/"):
                # Check if it's in the exception list
                is_exception = any(
                    re.match(pattern, path) for pattern in autumn_route_exceptions
                )
                
                if not is_exception:
                    # Add trailing slash to ALL /api/autumn/* routes by default
                    expected_path = f"{path}/"
                    scope = request.scope.copy()
                    scope["path"] = expected_path
                    scope["raw_path"] = expected_path.encode()
                    request._url = None  # Force URL rebuild
                    request.scope.update(scope)
        
        response = await call_next(request)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app startup and shutdown events.
    """
    # Startup
    yield
    
    # Shutdown - Clean shutdown handler for Autumn ASGI app
    await autumn_app.close()


app = FastAPI(
    lifespan=lifespan,
    servers=[
        {"url": "https://comfy-api-production.up.railway.app/api", "description": "Production server"},
        {
            "url": "https://staging.api.comfydeploy.com/api",
            "description": "Staging server",
        },
        {"url": "http://localhost:3011/api", "description": "Local development server"},
    ]
)

# Add trailing slash middleware to handle Autumn ASGI routing
app.add_middleware(TrailingSlashMiddleware)

api_router = APIRouter()  # Remove the prefix here
public_api_router = APIRouter()  # Remove the prefix here
basic_public_api_router = APIRouter()  # Remove the prefix here

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
    # if app.openapi_schema and with_code_samples:
    #     return app.openapi_schema
    
    # fetch_from_speakeasy = with_code_samples and os.getenv("ENV", "production").lower() == "production"
    # Disable fetching from speakeasy
    fetch_from_speakeasy = False
    
    if (fetch_from_speakeasy):
        # In development mode, fetch from Speakeasy
        import requests
        try:
            response = requests.get("https://spec.speakeasy.com/comfydeploy/comfydeploy/comfydeploy-api-with-code-samples")
            openapi_schema = response.json()
        except Exception as e:
            # logger.error(f"Failed to fetch Speakeasy schema: {e}")
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

def custom_simple_openapi(with_code_samples: bool = True):
    """Generate a simplified OpenAPI schema with only specific endpoints"""
    # Define allowed endpoints
    allowed_paths = [
        "/run/{run_id}",
        "/run/deployment/queue",
        "/run/{run_id}/cancel"
    ]
    
    # Get full schema (either from Speakeasy or generate locally)
    fetch_from_speakeasy = False #with_code_samples and os.getenv("ENV", "production").lower() == "production"
    
    try:
        if fetch_from_speakeasy:
            import requests
            response = requests.get("https://spec.speakeasy.com/comfydeploy/comfydeploy/comfydeploy-api-with-code-samples")
            full_schema = response.json()
        else:
            full_schema = get_openapi(
                title="ComfyDeploy API",
                version="V2",
                description=docs,
                routes=public_api_router.routes,
                servers=app.servers,
                webhooks=app.webhooks.routes,
            )
    except Exception as e:
        # Fallback to local generation if Speakeasy fails
        full_schema = get_openapi(
            title="ComfyDeploy API",
            version="V2",
            description=docs,
            routes=public_api_router.routes,
            servers=app.servers,
            webhooks=app.webhooks.routes,
        )
    
    # Create limited schema with only allowed paths
    # Handle case where 'paths' key doesn't exist
    full_paths = full_schema.get("paths", {})
    filtered_paths = {path: full_paths.get(path, {}) for path in allowed_paths if path in full_paths}
    
    return {
        **full_schema,
        "paths": filtered_paths,
        "components": {
            **full_schema.get("components", {}),
            "securitySchemes": {"Bearer": {"type": "http", "scheme": "bearer"}}
        },
        "security": [{"Bearer": []}]
    }

app.openapi = custom_simple_openapi

@app.get("/openapi.json", include_in_schema=False)
async def openapi_simple_json():
    """Return a simplified OpenAPI schema with only specific endpoints"""
    return JSONResponse(status_code=200, content=custom_simple_openapi())

@app.get("/openapi.json/with-no-code-samples", include_in_schema=False)
async def openapi_simple_json_no_samples():
    """Return a simplified OpenAPI schema without code samples"""
    return JSONResponse(status_code=200, content=custom_simple_openapi(with_code_samples=False))

@app.get("/", include_in_schema=False)
async def scalar_html_simple():
    """Return Scalar docs UI for simplified API endpoints"""
    return get_scalar_api_reference(
        openapi_url="/openapi.json",
        title="ComfyDeploy API",
        scalar_proxy_url="https://proxy.scalar.com",
        hide_models=True,
        servers=[{"url": server["url"]} for server in app.servers],
    )

