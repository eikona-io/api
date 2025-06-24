from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.openapi.utils import get_openapi
from fastapi import APIRouter
from scalar_fastapi import get_scalar_api_reference
from fastapi.responses import JSONResponse
import os


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


# app.openapi = custom_openapi

# @app.get("/", include_in_schema=False)
# async def scalar_html():
#     return get_scalar_api_reference(
#         openapi_url=app.openapi_url,
#         title=app.title,
#         scalar_proxy_url="https://proxy.scalar.com",
#         hide_models=True,
#         servers=[
#             {"url": server["url"]} for server in app.servers
#         ],  # Remove "/api" here
#     )
    
    
# @app.get("/openapi.json/with-no-code-samples", include_in_schema=False)
# async def openapi_json():
#     return JSONResponse(status_code=200, content=custom_openapi(with_code_samples=False))


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
    fetch_from_speakeasy = with_code_samples and os.getenv("ENV", "production").lower() == "production"
    
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
