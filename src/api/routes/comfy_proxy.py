from fastapi import HTTPException, APIRouter, Request, Depends
from fastapi.responses import Response
import logging
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
import os

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Comfy Proxy"])

COMFY_API_KEY = os.getenv("COMFY_API_KEY")

def check_enterprise_plan(request: Request):
    """Check if user is on enterprise plan"""
    if not hasattr(request.state, 'current_user') or request.state.current_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    plan = request.state.current_user.get("plan")
    if not plan:
        raise HTTPException(status_code=403, detail="No plan found")
    
    # Check for enterprise plan variants
    enterprise_plans = ["enterprise", "business", "business_monthly", "business_yearly"]
    if plan not in enterprise_plans:
        raise HTTPException(
            status_code=403, 
            detail="Enterprise plan required to access Comfy.org API proxy"
        )
    
    return True

@router.api_route("/comfy-org/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_to_comfy_org(
    request: Request,
    path: str,
    db: AsyncSession = Depends(get_db),
):
    """Proxy all requests to https://api.comfy.org/ with enterprise plan validation"""
    
    # Check enterprise plan
    check_enterprise_plan(request)
    
    # Build target URL
    target_url = f"https://api.comfy.org/{path}"
    
    # Get query parameters
    query_params = dict(request.query_params)
    
    # Get request body if exists
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
        except Exception as e:
            logger.error(f"Error reading request body: {str(e)}")
            body = None
    
    # Prepare headers (exclude host and connection headers)
    headers = {}
    for key, value in request.headers.items():
        if key.lower() not in ["host", "connection", "content-length"]:
            headers[key] = value
            
    headers["x-api-key"] = COMFY_API_KEY
    
    try:
        async with httpx.AsyncClient() as client:
            # Make the proxied request
            response = await client.request(
                method=request.method,
                url=target_url,
                params=query_params,
                content=body,
                headers=headers,
                follow_redirects=True,
                timeout=30.0
            )
            
            # Prepare response headers (exclude some that shouldn't be proxied)
            response_headers = {}
            for key, value in response.headers.items():
                if key.lower() not in ["content-encoding", "transfer-encoding", "connection"]:
                    response_headers[key] = value
            
            # Return the proxied response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type")
            )
            
    except httpx.TimeoutException:
        logger.error(f"Timeout when proxying to {target_url}")
        raise HTTPException(status_code=504, detail="Gateway timeout when accessing Comfy.org API")
    except httpx.RequestError as e:
        logger.error(f"Request error when proxying to {target_url}: {str(e)}")
        raise HTTPException(status_code=502, detail="Bad gateway when accessing Comfy.org API")
    except Exception as e:
        logger.error(f"Unexpected error when proxying to {target_url}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error when proxying request")