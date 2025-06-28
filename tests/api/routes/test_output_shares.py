from conftest import get_test_client
import pytest
import uuid
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.asyncio
async def test_create_output_share_success(app, paid_user, test_run_deployment_sync_public):
    """Test successful creation of output share"""
    run_id = test_run_deployment_sync_public
    
    async with get_test_client(app, paid_user) as client:
        response = await client.post("/output-shares", json={
            "run_id": str(run_id),
            "output_id": str(run_id),
            "visibility": "public"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == str(run_id)
        assert data["output_id"] == str(run_id)
        assert data["visibility"] == "public"
        assert "share_slug" in data
        assert len(data["share_slug"]) == 8


@pytest.mark.asyncio
async def test_create_output_share_default_visibility(app, paid_user, test_run_deployment_sync_public):
    """Test output share creation with default visibility"""
    run_id = test_run_deployment_sync_public
    
    async with get_test_client(app, paid_user) as client:
        response = await client.post("/output-shares", json={
            "run_id": str(run_id),
            "output_id": str(run_id)
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["visibility"] == "link-only"


@pytest.mark.asyncio
async def test_create_output_share_nonexistent_run(app, paid_user):
    """Test creating output share for nonexistent run"""
    async with get_test_client(app, paid_user) as client:
        response = await client.post("/output-shares", json={
            "run_id": str(uuid.uuid4()),
            "output_id": str(uuid.uuid4())
        })
        
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_output_share_unauthorized_run(app, paid_user_2, test_run_deployment_sync_public):
    """Test creating output share for run owned by different user"""
    run_id = test_run_deployment_sync_public
    
    async with get_test_client(app, paid_user_2) as client:
        response = await client.post("/output-shares", json={
            "run_id": str(run_id),
            "output_id": str(run_id)
        })
        
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_output_share_free_user_forbidden(app, free_user, test_run_deployment_sync_public):
    """Test free user cannot create output shares"""
    run_id = test_run_deployment_sync_public
    
    async with get_test_client(app, free_user) as client:
        response = await client.post("/output-shares", json={
            "run_id": str(run_id),
            "output_id": str(run_id)
        })
        
        assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_output_shares_success(app, paid_user, test_run_deployment_sync_public):
    """Test listing output shares"""
    run_id = test_run_deployment_sync_public
    
    async with get_test_client(app, paid_user) as client:
        share_response = await client.post("/output-shares", json={
            "run_id": str(run_id),
            "output_id": str(run_id)
        })
        assert share_response.status_code == 200
        
        list_response = await client.get("/output-shares")
        assert list_response.status_code == 200
        shares = list_response.json()
        assert len(shares) >= 1
        assert any(share["run_id"] == str(run_id) for share in shares)


@pytest.mark.asyncio
async def test_list_output_shares_empty(app, paid_user):
    """Test listing when user has no output shares"""
    async with get_test_client(app, paid_user) as client:
        response = await client.get("/output-shares")
        assert response.status_code == 200
        shares = response.json()
        assert isinstance(shares, list)


@pytest.mark.asyncio
async def test_list_output_shares_isolation(app, paid_user):
    """Test users can only see their own output shares"""
    async with get_test_client(app, paid_user) as client:
        response = await client.get("/output-shares")
        assert response.status_code == 200
        shares = response.json()
        assert isinstance(shares, list)


@pytest.mark.asyncio
async def test_get_shared_outputs_success(app, paid_user, test_run_deployment_sync_public):
    """Test retrieving shared outputs by slug"""
    run_id = test_run_deployment_sync_public
    
    async with get_test_client(app, paid_user) as client:
        share_response = await client.post("/output-shares", json={
            "run_id": str(run_id),
            "output_id": str(run_id),
            "visibility": "public"
        })
        assert share_response.status_code == 200
        share_slug = share_response.json()["share_slug"]
        
        get_response = await client.get(f"/output-shares/{share_slug}")
        assert get_response.status_code == 200
        data = get_response.json()
        assert "share" in data
        assert "run" in data
        assert data["share"]["share_slug"] == share_slug


@pytest.mark.asyncio
async def test_get_shared_outputs_nonexistent_slug(app, paid_user):
    """Test retrieving nonexistent shared outputs"""
    async with get_test_client(app, paid_user) as client:
        response = await client.get("/output-shares/nonexistent")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_shared_outputs_visibility_controls(app, test_free_user, test_run_deployment_sync_public):
    """Test visibility controls for shared outputs"""
    run_id = test_run_deployment_sync_public
    
    visibility_levels = ["link-only", "public", "public-in-org"]
    
    async with get_test_client(app, test_free_user) as client:
        created_shares = []
        
        for visibility in visibility_levels:
            share_response = await client.post("/output-shares", json={
                "run_id": str(run_id),
                "output_id": str(run_id),
                "visibility": visibility
            })
            assert share_response.status_code == 200
            created_shares.append(share_response.json())
        
        for share in created_shares:
            get_response = await client.get(f"/output-shares/{share['share_slug']}")
            assert get_response.status_code == 200


@pytest.mark.asyncio
async def test_delete_output_share_success(app, paid_user, test_run_deployment_sync_public):
    """Test successful deletion of output share"""
    run_id = test_run_deployment_sync_public
    
    async with get_test_client(app, paid_user) as client:
        share_response = await client.post("/output-shares", json={
            "run_id": str(run_id),
            "output_id": str(run_id)
        })
        assert share_response.status_code == 200
        share_id = share_response.json()["id"]
        
        delete_response = await client.delete(f"/output-shares/{share_id}")
        assert delete_response.status_code == 200
        
        list_response = await client.get("/output-shares")
        shares = list_response.json()
        assert not any(share["id"] == share_id for share in shares)


@pytest.mark.asyncio
async def test_delete_output_share_nonexistent(app, paid_user):
    """Test deleting nonexistent output share"""
    async with get_test_client(app, paid_user) as client:
        response = await client.delete(f"/output-shares/{uuid.uuid4()}")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_output_share_unauthorized(app, paid_user):
    """Test unauthorized deletion of output share"""
    async with get_test_client(app, paid_user) as client:
        response = await client.delete(f"/output-shares/{uuid.uuid4()}")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_output_share_org_isolation(app, paid_user, paid_user_2, test_run_deployment_sync_public):
    """Test org-level isolation of output shares"""
    run_id = test_run_deployment_sync_public
    
    async with get_test_client(app, paid_user) as client:
        share_response = await client.post("/output-shares", json={
            "run_id": str(run_id),
            "output_id": str(run_id)
        })
        assert share_response.status_code == 200
    
    async with get_test_client(app, paid_user_2) as client:
        list_response = await client.get("/output-shares")
        shares = list_response.json()
        assert len(shares) == 0


@pytest.mark.asyncio
async def test_create_output_share_empty_output_ids(app, paid_user, test_run_deployment_sync_public):
    """Test creating output share with empty output IDs"""
    run_id = test_run_deployment_sync_public
    
    async with get_test_client(app, paid_user) as client:
        response = await client.post("/output-shares", json={
            "run_id": str(run_id),
            "output_id": str(run_id)
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["output_id"] == str(run_id)


@pytest.mark.asyncio
async def test_create_output_share_invalid_visibility(app, paid_user, test_run_deployment_sync_public):
    """Test creating output share with invalid visibility"""
    run_id = test_run_deployment_sync_public
    
    async with get_test_client(app, paid_user) as client:
        response = await client.post("/output-shares", json={
            "run_id": str(run_id),
            "output_id": str(run_id),
            "visibility": "invalid_visibility"
        })
        
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_slug_uniqueness(app, test_free_user, test_run_deployment_sync_public):
    """Test that share slugs are unique"""
    run_id = test_run_deployment_sync_public
    
    created_slugs = set()
    async with get_test_client(app, test_free_user) as client:
        for i in range(5):
            response = await client.post("/output-shares", json={
                "run_id": str(run_id),
                "output_id": str(run_id)
            })
            assert response.status_code == 200
            share_slug = response.json()["share_slug"]
            assert share_slug not in created_slugs
            created_slugs.add(share_slug)
        
        assert len(created_slugs) == 5
