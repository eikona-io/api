import pytest
from httpx import AsyncClient
from conftest import get_test_client, get_db_context
from api.models import User
from api.routes.utils import generate_machine_token, generate_persistent_token
from uuid import uuid4
import pytest_asyncio
from upstash_redis import Redis
import os
import json
from datetime import datetime

redis_url = os.getenv("UPSTASH_REDIS_META_REST_URL")
redis_token = os.getenv("UPSTASH_REDIS_META_REST_TOKEN")

@pytest_asyncio.fixture(scope="session")
async def machine_token_user():
    """Create a test user for machine token tests"""
    user_id = str(uuid4())

    redis = Redis(url=redis_url, token=redis_token)
    data = {
        "data": {
            "plans": ["business_monthly"],
            "names": [],
            "prices": [],
            "amount": [],
            "charges": [],
            "cancel_at_period_end": False,
            "canceled_at": None,
            "payment_issue": False,
            "payment_issue_reason": "",
        },
        "version": "1.0",
        "timestamp": int(datetime.now().timestamp()),
    }
    redis.set(f"plan:{user_id}", json.dumps(data))

    async with get_db_context() as db:
        user = User(
            id=user_id,
            username="machine_token_user",
            name="Machine Token User",
        )
        db.add(user)
    yield user


async def get_machine_token_client(app, user, org_id=None):
    """Helper function to create a client with machine token"""
    machine_token = generate_machine_token(user.id, org_id)
    client = AsyncClient(
        base_url=app + "/api",
        headers={"Authorization": f"Bearer {machine_token}"},
        timeout=120.0,
    )
    try:
        yield client
    finally:
        await client.aclose()


async def get_regular_token_client(app, user, org_id=None):
    """Helper function to create a client with regular user token"""
    user_token = generate_persistent_token(user.id, org_id)
    client = AsyncClient(
        base_url=app + "/api",
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=120.0,
    )
    try:
        yield client
    finally:
        await client.aclose()


# Test machine token access to ALLOWED endpoints
@pytest.mark.asyncio
async def test_machine_token_allowed_endpoints(app, machine_token_user):
    """Test that machine tokens can access allowed endpoints"""
    
    async with get_machine_token_client(app, machine_token_user) as client:
        # Test GPU event endpoint (should be allowed)
        gpu_event_data = {
            "event_type": "test",
            "machine_id": str(uuid4()),
            "data": {"test": "data"}
        }
        response = await client.post("/gpu_event", json=gpu_event_data)
        # Note: This might fail due to validation, but it should NOT fail with 403
        assert response.status_code != 403, "Machine token should be allowed to access /api/gpu_event"
        
        # Test update-run endpoint (should be allowed)
        update_run_data = {
            "run_id": str(uuid4()),
            "status": "running"
        }
        response = await client.post("/update-run", json=update_run_data)
        # Note: This might fail due to validation, but it should NOT fail with 403
        assert response.status_code != 403, "Machine token should be allowed to access /api/update-run"
        
        # Test file-upload endpoint (should be allowed)
        response = await client.get("/file-upload")
        # Note: This might fail due to missing parameters, but it should NOT fail with 403
        assert response.status_code != 403, "Machine token should be allowed to access /api/file-upload"
        
        # Test machine-built endpoint (should be allowed)
        machine_built_data = {
            "machine_id": str(uuid4()),
            "status": "built"
        }
        response = await client.post("/machine-built", json=machine_built_data)
        # Note: This might fail due to validation, but it should NOT fail with 403
        assert response.status_code != 403, "Machine token should be allowed to access /api/machine-built"


# Test machine token DENIED access to restricted endpoints
@pytest.mark.asyncio 
async def test_machine_token_denied_endpoints(app, machine_token_user):
    """Test that machine tokens are denied access to restricted endpoints"""
    
    async with get_machine_token_client(app, machine_token_user) as client:
        
        # Test workflow endpoints (should be denied)
        workflow_data = {
            "name": "test-workflow",
            "workflow_json": "{}",
            "workflow_api": "{}",
            "machine_id": str(uuid4())
        }
        response = await client.post("/workflow", json=workflow_data)
        assert response.status_code == 403, "Machine token should be denied access to /api/workflow"
        
        response = await client.get("/workflow")
        assert response.status_code == 403, "Machine token should be denied access to /api/workflow"
        
        # Test deployment endpoints (should be denied)
        deployment_data = {
            "workflow_id": str(uuid4()),
            "machine_id": str(uuid4()),
            "environment": "production"
        }
        response = await client.post("/deployment", json=deployment_data)
        assert response.status_code == 403, "Machine token should be denied access to /api/deployment"
        
        response = await client.get("/deployment")
        assert response.status_code == 403, "Machine token should be denied access to /api/deployment"
        
        # Test machine management endpoints (should be denied)
        machine_data = {
            "name": "test-machine",
            "type": "classic",
            "endpoint": "http://localhost:8188"
        }
        response = await client.post("/machine/custom", json=machine_data)
        assert response.status_code == 403, "Machine token should be denied access to /api/machine/custom"
        
        response = await client.post("/machine/serverless", json=machine_data)
        assert response.status_code == 403, "Machine token should be denied access to /api/machine/serverless"
        
        # Test platform endpoints (should be denied)
        response = await client.get("/platform/subscription")
        assert response.status_code == 403, "Machine token should be denied access to /api/platform/subscription"
        
        # Test admin endpoints (should be denied)
        response = await client.get("/admin/users")
        assert response.status_code == 403, "Machine token should be denied access to /api/admin/users"
        
        # Test session endpoints (should be denied)
        response = await client.get("/session")
        assert response.status_code == 403, "Machine token should be denied access to /api/session"
        
        # Test run endpoints (should be denied)
        run_data = {
            "deployment_id": str(uuid4()),
            "inputs": {}
        }
        response = await client.post("/run/deployment", json=run_data)
        assert response.status_code == 403, "Machine token should be denied access to /api/run/deployment"


# Test comfy-org glob pattern (should be allowed)
@pytest.mark.asyncio
async def test_machine_token_comfy_org_endpoints(app, machine_token_user):
    """Test that machine tokens can access comfy-org endpoints with glob pattern"""
    
    async with get_machine_token_client(app, machine_token_user) as client:
        # Test base comfy-org endpoint
        response = await client.get("/comfy-org")
        assert response.status_code != 403, "Machine token should be allowed to access /api/comfy-org"
        
        # Test comfy-org sub-endpoints
        response = await client.get("/comfy-org/models")
        assert response.status_code != 403, "Machine token should be allowed to access /api/comfy-org/models"
        
        response = await client.get("/comfy-org/anything")
        assert response.status_code != 403, "Machine token should be allowed to access /api/comfy-org/anything"


# Test proxy glob pattern (should be allowed)
@pytest.mark.asyncio
async def test_machine_token_proxy_endpoints(app, machine_token_user):
    """Test that machine tokens can access proxy endpoints with glob pattern"""
    
    async with get_machine_token_client(app, machine_token_user) as client:
        # Test base proxy endpoint
        response = await client.get("/proxy")
        assert response.status_code != 403, "Machine token should be allowed to access /proxy"
        
        # Test proxy sub-endpoints
        response = await client.get("/proxy/some/path")
        assert response.status_code != 403, "Machine token should be allowed to access /proxy/some/path"


# Test comparison with regular user token
@pytest.mark.asyncio
async def test_regular_token_vs_machine_token_access(app, machine_token_user):
    """Test that regular user tokens have broader access than machine tokens"""
    
    # Test with regular user token - should have access to workflow endpoints
    async with get_regular_token_client(app, machine_token_user) as client:
        response = await client.get("/workflow")
        assert response.status_code != 403, "Regular user token should be allowed to access /api/workflow"
        
        response = await client.get("/deployment")
        assert response.status_code != 403, "Regular user token should be allowed to access /api/deployment"
    
    # Test with machine token - should be denied access to the same endpoints
    async with get_machine_token_client(app, machine_token_user) as client:
        response = await client.get("/workflow")
        assert response.status_code == 403, "Machine token should be denied access to /api/workflow"
        
        response = await client.get("/deployment")
        assert response.status_code == 403, "Machine token should be denied access to /api/deployment"


# Test machine token with organization
@pytest.mark.asyncio 
async def test_machine_token_with_org(app, machine_token_user):
    """Test that machine tokens work properly with organization context"""
    
    org_id = str(uuid4())
    
    async with get_machine_token_client(app, machine_token_user, org_id) as client:
        # Test that org-scoped machine token still respects endpoint restrictions
        response = await client.get("/workflow")
        assert response.status_code == 403, "Org-scoped machine token should still be denied access to /api/workflow"
        
        # Test that org-scoped machine token can access allowed endpoints
        gpu_event_data = {
            "event_type": "test",
            "machine_id": str(uuid4()),
            "data": {"test": "data"}
        }
        response = await client.post("/gpu_event", json=gpu_event_data)
        assert response.status_code != 403, "Org-scoped machine token should be allowed to access /api/gpu_event"


# Test invalid/malformed machine token
@pytest.mark.asyncio
async def test_invalid_machine_token(app):
    """Test that invalid machine tokens are properly rejected"""
    
    # Test with invalid token
    client = AsyncClient(
        base_url=app + "/api",
        headers={"Authorization": "Bearer invalid_token"},
        timeout=120.0,
    )
    
    try:
        response = await client.get("/gpu_event")
        assert response.status_code == 401, "Invalid token should result in 401 Unauthorized"
        
        response = await client.get("/workflow")
        assert response.status_code == 401, "Invalid token should result in 401 Unauthorized"
    finally:
        await client.aclose()


# Test no authorization header
@pytest.mark.asyncio
async def test_no_authorization_header(app):
    """Test that requests without authorization are properly rejected"""
    
    client = AsyncClient(
        base_url=app + "/api",
        timeout=120.0,
    )
    
    try:
        response = await client.get("/gpu_event")
        assert response.status_code == 401, "Request without auth should result in 401 Unauthorized"
        
        response = await client.get("/workflow")
        assert response.status_code == 401, "Request without auth should result in 401 Unauthorized"
    finally:
        await client.aclose()


# Test edge cases with glob patterns
@pytest.mark.asyncio
async def test_machine_token_glob_pattern_edge_cases(app, machine_token_user):
    """Test edge cases with glob pattern matching"""
    
    async with get_machine_token_client(app, machine_token_user) as client:
        # Test that patterns don't accidentally allow broader access
        response = await client.get("/api-comfy-org")  # Should NOT match /api/comfy-org/*
        assert response.status_code == 403, "Machine token should not match similar but different paths"
        
        # Test that exact prefix matching works
        response = await client.get("/comfy-org-test")  # Should NOT match /api/comfy-org/*
        assert response.status_code == 403, "Machine token should not match paths that don't start with the exact pattern" 