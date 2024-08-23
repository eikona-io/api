import pytest
from httpx import ASGITransport, AsyncClient
from .__init__ import app

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidXNlcl8yWkE2dnVLRDNJSlhqdTE2b0pWUUdMQmNXd2ciLCJvcmdfaWQiOiJvcmdfMmJXUTFGb1dDM1dybzM5MVR1cmtlVkc3N3BDIiwiaWF0IjoxNzIwMTM5NjMyfQ.BBSt5kWJRPDAwNx2Sk_EJU5XRoUJpmTv1hLZvuf1r-M"
run_id = "01babcd2-52a8-4fb6-82b3-949a4cdc94d8"

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
async def async_client():
    client = await AsyncClient(base_url="http://test", transport=ASGITransport(app=app))
    return client

# @pytest.mark.asyncio
# async def test_hello_endpoint(async_client):
#     response = await async_client.get("/hello")
#     assert response.status_code == 200
#     assert response.json() == {"message": "Hello from api!"}


async def test_run_endpoint():
    async with AsyncClient(
        base_url="http://test", transport=ASGITransport(app=app)
    ) as client:
        response = await client.get(
            f"/api/run?run_id={run_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert "workflow_version_id" in response.json()


async def test_run_endpoint_without_token():
    async with AsyncClient(
        base_url="http://test", transport=ASGITransport(app=app)
    ) as client:
        response = await client.get(f"/api/run?run_id={run_id}")
        assert response.status_code == 401
        assert "detail" in response.json()
        assert response.json()["detail"] == "Invalid or missing token"


async def test_run_endpoint_with_revoked_token():
    revoked_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidXNlcl8yWkE2dnVLRDNJSlhqdTE2b0pWUUdMQmNXd2ciLCJvcmdfaWQiOiJvcmdfMmJXUTFGb1dDM1dybzM5MVR1cmtlVkc3N3BDIiwiaWF0IjoxNzI0MzYyNzE3fQ.lcIiYjxngXwMhPTxHozYazzW-jso_-QPNCsL0fJw24g"
    async with AsyncClient(
        base_url="http://test", transport=ASGITransport(app=app)
    ) as client:
        response = await client.get(
            f"/api/run?run_id={run_id}",
            headers={"Authorization": f"Bearer {revoked_token}"}
        )
    assert response.status_code == 401
    assert "detail" in response.json()
    assert response.json()["detail"] == "Invalid or expired token"
