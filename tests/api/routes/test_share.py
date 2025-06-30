import pytest
from conftest import get_test_client


@pytest.mark.asyncio
async def test_share_output_prevent_duplicate(
    app, test_free_user, test_run_deployment_sync_public_with_output
):
    run_id, output_id = test_run_deployment_sync_public_with_output
    async with get_test_client(app, test_free_user) as client:
        response = await client.post(
            "/share/output",
            json={"run_id": run_id, "output_id": output_id},
        )
        assert response.status_code == 200

        # Try sharing the same output again from the same user
        response_dup = await client.post(
            "/share/output",
            json={"run_id": run_id, "output_id": output_id},
        )
        assert response_dup.status_code == 200
        assert response_dup.json()["id"] == response.json()["id"]


@pytest.mark.asyncio
async def test_share_output_pagination(
    app, test_free_user, test_run_deployment_sync_public_with_output
):
    """Verify pagination parameters limit and offset."""
    run_id, output_id = test_run_deployment_sync_public_with_output
    async with get_test_client(app, test_free_user) as client:
        # ensure share exists
        await client.post(
            "/share/output",
            json={"run_id": run_id, "output_id": output_id},
        )

        # request first item with limit
        resp = await client.get("/share/output", params={"limit": 1, "offset": 0})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        # offset beyond available items should return empty list
        resp = await client.get("/share/output", params={"limit": 20, "offset": 1})
        assert resp.status_code == 200
        assert resp.json() == []
