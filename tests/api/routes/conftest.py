import pytest  # noqa: E402
from httpx import AsyncClient  # noqa: E402
import pytest_asyncio  # noqa: E402
from uuid import uuid4  # noqa: E402
from upstash_redis import Redis
from datetime import datetime
from contextlib import asynccontextmanager  # noqa: E402
from api.models import User  # noqa: E402
import os
import json
from api.routes.utils import generate_persistent_token, generate_machine_token  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession  # noqa: E402
from sqlalchemy.pool import AsyncAdaptedQueuePool  # noqa: E402
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv("UPSTASH_REDIS_META_REST_URL")
redis_token = os.getenv("UPSTASH_REDIS_META_REST_TOKEN")
# Use environment variables for database connection
DATABASE_URL = os.getenv("DATABASE_URL")

# Ensure the URL uses the asyncpg dialect
if DATABASE_URL and not DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")



# Configure engine with larger pool size and longer timeout
engine = create_async_engine(
    DATABASE_URL,
    poolclass=AsyncAdaptedQueuePool,
    pool_size=20,  # Increased from default 5
    max_overflow=30,  # Increased from default 10
    pool_timeout=60,  # Increased from default 30
    pool_pre_ping=True,  # Enable connection health checks
    pool_recycle=3600,  # Recycle connections after 1 hour
)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def get_db_context():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_test_client(app, user):
    """Helper function to create a new client instance with async context manager support"""
    if user is None:
        client = AsyncClient(
            base_url=app + "/api",
            timeout=120.0,  # 30 seconds timeout for all operations
        )
    else:
        api_key = generate_persistent_token(user.id, None)
        client = AsyncClient(
            base_url=app + "/api",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120.0,  # 30 seconds timeout for all operations
        )
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def test_client(app, user):
    """Fixture to provide an authenticated test client"""
    api_key = generate_persistent_token(user.id, None)
    client = AsyncClient(
        base_url=app + "/api",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120.0,
    )
    yield client
    await client.aclose()


@asynccontextmanager
async def get_machine_test_client(app, user, org_id=None):
    """Helper function to create a new client instance with machine token"""
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


basic_workflow_json = """
{
  "last_node_id": 20,
  "last_link_id": 13,
  "nodes": [
    {
      "id": 17,
      "type": "SaveImage",
      "pos": {
        "0": 999.4330444335938,
        "1": 372.1318664550781
      },
      "size": {
        "0": 315,
        "1": 58
      },
      "flags": {},
      "order": 1,
      "mode": 0,
      "inputs": [
        {
          "link": 13,
          "name": "images",
          "type": "IMAGE"
        }
      ],
      "outputs": [],
      "properties": {},
      "widgets_values": [
        "ComfyUI"
      ]
    },
    {
      "id": 16,
      "type": "ComfyUIDeployExternalImage",
      "pos": {
        "0": 387,
        "1": 424
      },
      "size": {
        "0": 390.5999755859375,
        "1": 366
      },
      "flags": {},
      "order": 0,
      "mode": 0,
      "inputs": [
        {
          "link": null,
          "name": "default_value",
          "type": "IMAGE",
          "shape": 7
        }
      ],
      "outputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "links": [
            13
          ],
          "slot_index": 0
        }
      ],
      "properties": {
        "Node name for S&R": "ComfyUIDeployExternalImage"
      },
      "widgets_values": [
        "https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/assets/img_bRFqDVG5VG87N29W.png",
        "",
        "",
        "https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/assets/img_bRFqDVG5VG87N29W.png",
        ""
      ]
    }
  ],
  "links": [
    [
      13,
      16,
      0,
      17,
      0,
      "IMAGE"
    ]
  ],
  "groups": [],
  "config": {},
  "extra": {
    "ds": {
      "scale": 1.3109994191499965,
      "offset": {
        "0": -242.70708454566912,
        "1": -115.02119499557512
      }
    },
    "node_versions": {
      "comfy-core": "v0.2.4",
      "comfyui-deploy": "4073a43d3d04f6659acc7954f79a4fa7d83a3867"
    }
  },
  "version": 0.4
}
"""
basic_workflow_api_json = """
{
  "16": {
    "inputs": {
      "input_id": "https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/assets/img_bRFqDVG5VG87N29W.png",
      "display_name": "",
      "description": "",
      "default_value_url": "https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/assets/img_bRFqDVG5VG87N29W.png"
    },
    "class_type": "ComfyUIDeployExternalImage"
  },
  "17": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "16",
        0
      ]
    },
    "class_type": "SaveImage"
  }
}
"""



@pytest_asyncio.fixture(scope="session")
async def paid_user():
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
    print("redis set", redis.get(f"plan:{user_id}"))

    async with get_db_context() as db:
        user = User(
            id=user_id,
            username="business_user",
            name="Business User",
        )
        db.add(user)
    yield user

@pytest_asyncio.fixture(scope="session")
async def paid_user_2():
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
    print("redis set", redis.get(f"plan:{user_id}"))

    async with get_db_context() as db:
        user = User(
            id=user_id,
            username="business_user",
            name="Business User",
        )
        db.add(user)
    yield user



@pytest_asyncio.fixture(scope="session")
async def free_user():
    user_id = str(uuid4())

    redis = Redis(url=redis_url, token=redis_token)
    data = {
        "plans": [],
        "names": [],
        "prices": [],
        "amount": [],
        "charges": [],
        "cancel_at_period_end": False,
        "canceled_at": None,
        "payment_issue": False,
        "payment_issue_reason": "",
    }
    redis.set(f"plan:{user_id}", json.dumps(data))
    print("redis set", redis.get(f"plan:{user_id}"))

    async with get_db_context() as db:
        user = User(
            id=user_id,
            username="free_user",
            name="Free User",
        )
        db.add(user)
    yield user


@pytest_asyncio.fixture(scope="session")
async def app():
    import uvicorn
    import requests
    from time import sleep

    # Start the server in a separate thread/process
    import multiprocessing

    def run_server():
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

    # Start server process
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()

    # Wait for server to be ready
    max_retries = 30  # Maximum number of retries
    retry_delay = 0.5  # Delay between retries in seconds

    print("Waiting for server to be ready")

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries}")
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                break
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise RuntimeError("Failed to start server after maximum retries")
            sleep(retry_delay)

    yield "http://localhost:8000"

    print("Stopping server")
    # Cleanup: stop the server
    server_process.terminate()
    server_process.join()


@pytest_asyncio.fixture(scope="session")
async def test_serverless_machine(app, paid_user):
    machine_data = {
        "name": "test-serverless-machine-paid",
        "gpu": "CPU",
        "wait_for_build": True,
    }

    async with get_test_client(app, paid_user) as client:
        response = await client.post("/machine/serverless", json=machine_data)
        assert response.status_code == 200
        machine_id = response.json()["id"]

        yield machine_id

        delete_response = await client.delete(f"/machine/{machine_id}?force=true")
        assert delete_response.status_code == 200

@pytest_asyncio.fixture(scope="session")
async def test_create_workflow_deployment(app, paid_user, test_serverless_machine):
    """Test creating a workflow and deployment"""
    async with get_test_client(app, paid_user) as client:
        workflow_data = {
            "name": "test-workflow",
            "workflow_json": basic_workflow_json,
            "workflow_api": basic_workflow_api_json,
            "machine_id": test_serverless_machine,
        }
        response = await client.post("/workflow", json=workflow_data)
        assert response.status_code == 200, (
            f"Workflow creation failed with response: {response.text}"
        )
        workflow_id = response.json()["workflow_id"]

        response = await client.get(f"/workflow/{workflow_id}/versions")
        assert response.status_code == 200, (
            f"Getting workflow versions failed with response: {response.text}"
        )
        workflow_version_id = response.json()[0]["id"]

        deployment_data = {
            "workflow_id": workflow_id,
            "workflow_version_id": workflow_version_id,
            "machine_id": test_serverless_machine,
            "environment": "production",
        }
        print(f"Deployment data: {deployment_data}")
        response = await client.post("/deployment", json=deployment_data)
        if response.status_code != 200:
            print(f"Deployment creation failed with status {response.status_code}")
            print(f"Response body: {response.text}")
            raise AssertionError(f"Deployment creation failed: {response.text}")
        deployment_id = response.json()["id"]

        yield deployment_id

@pytest_asyncio.fixture(scope="session")
async def test_create_workflow_deployment_public(app, paid_user, test_serverless_machine):
    """Test creating a workflow and deployment"""
    async with get_test_client(app, paid_user) as client:
        workflow_data = {
            "name": "test-workflow-public",
            "workflow_json": basic_workflow_json,
            "workflow_api": basic_workflow_api_json,
            "machine_id": test_serverless_machine,
        }
        response = await client.post("/workflow", json=workflow_data)
        assert response.status_code == 200, (
            f"Workflow creation failed with response: {response.text}"
        )
        workflow_id = response.json()["workflow_id"]

        response = await client.get(f"/workflow/{workflow_id}/versions")
        assert response.status_code == 200, (
            f"Getting workflow versions failed with response: {response.text}"
        )
        workflow_version_id = response.json()[0]["id"]

        deployment_data = {
            "workflow_id": workflow_id,
            "workflow_version_id": workflow_version_id,
            "machine_id": test_serverless_machine,
            "environment": "public-share",
        }
        print(f"Deployment data: {deployment_data}")
        response = await client.post("/deployment", json=deployment_data)
        if response.status_code != 200:
            print(f"Deployment creation failed with status {response.status_code}")
            print(f"Response body: {response.text}")
            raise AssertionError(f"Deployment creation failed: {response.text}")
        deployment_id = response.json()["id"]

        yield deployment_id


basic_workflow_json_output_id = """
{"extra":{"ds":{"scale":1,"offset":[-134.83461235393543,-31.966476026948783]}},"links":[[14,16,0,18,0,"IMAGE"]],"nodes":[{"id":16,"pos":[337.5181884765625,284.7711486816406],"mode":0,"size":[390.5999755859375,366],"type":"ComfyUIDeployExternalImage","flags":{},"order":0,"inputs":[{"link":null,"name":"default_value","type":"IMAGE","shape":7}],"outputs":[{"name":"image","type":"IMAGE","links":[14],"slot_index":0}],"properties":{"Node name for S&R":"ComfyUIDeployExternalImage"},"widgets_values":["input_image","","","https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/assets/img_bRFqDVG5VG87N29W.png",""]},{"id":18,"pos":[893.0611572265625,382.5367736816406],"mode":0,"size":[327.5999755859375,130],"type":"ComfyDeployOutputImage","flags":{},"order":1,"inputs":[{"link":14,"name":"images","type":"IMAGE"}],"outputs":[],"properties":{"Node name for S&R":"ComfyDeployOutputImage"},"widgets_values":["ComfyUI","webp",80,"my_image"]}],"config":{},"groups":[],"version":0.4,"last_link_id":14,"last_node_id":18}
"""
basic_workflow_api_json_output_id = """
{"16":{"_meta":{"title":"External Image (ComfyUI Deploy)"},"inputs":{"input_id":"input_image","description":"","display_name":"","default_value_url":"https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/assets/img_bRFqDVG5VG87N29W.png"},"class_type":"ComfyUIDeployExternalImage"},"18":{"_meta":{"title":"Image Output (ComfyDeploy)"},"inputs":{"images":["16",0],"quality":80,"file_type":"webp","output_id":"my_image","filename_prefix":"ComfyUI"},"class_type":"ComfyDeployOutputImage"}}
"""

@pytest_asyncio.fixture(scope="session")
async def test_create_workflow_deployment_output_id(app, paid_user, test_serverless_machine):
    """Test creating a workflow and deployment"""
    async with get_test_client(app, paid_user) as client:
        workflow_data = {
            "name": "test-workflow-output-id",
            "workflow_json": basic_workflow_json_output_id,
            "workflow_api": basic_workflow_api_json_output_id,
            "machine_id": test_serverless_machine,
        }
        response = await client.post("/workflow", json=workflow_data)
        assert response.status_code == 200, (
            f"Workflow creation failed with response: {response.text}"
        )
        workflow_id = response.json()["workflow_id"]

        response = await client.get(f"/workflow/{workflow_id}/versions")
        assert response.status_code == 200, (
            f"Getting workflow versions failed with response: {response.text}"
        )
        workflow_version_id = response.json()[0]["id"]

        deployment_data = {
            "workflow_id": workflow_id,
            "workflow_version_id": workflow_version_id,
            "machine_id": test_serverless_machine,
            "environment": "production",
        }
        print(f"Deployment data: {deployment_data}")
        response = await client.post("/deployment", json=deployment_data)
        if response.status_code != 200:
            print(f"Deployment creation failed with status {response.status_code}")
            print(f"Response body: {response.text}")
            raise AssertionError(f"Deployment creation failed: {response.text}")
        deployment_id = response.json()["id"]

        yield deployment_id


@pytest_asyncio.fixture(scope="session")
async def test_run_deployment_sync_public(app, test_free_user, test_create_workflow_deployment_public):
    """Test running a deployment"""
    async with get_test_client(app, test_free_user) as client:
        deployment_id = test_create_workflow_deployment_public
        response = await client.post(
            "/run/deployment/sync", json={"deployment_id": deployment_id}
        )
        assert response.status_code == 200
        run_id = response.json()[0]["run_id"]
        assert run_id is not None
        yield run_id


@pytest_asyncio.fixture(scope="session")
async def test_free_user():
    """Fixture for a test user with free plan"""
    async with get_db_context() as db:
        user = User(
            id=str(uuid4()),
            username="test_free_user",
            name="Test Free User",
            # The plan will be handled by the backend based on user's subscription
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    yield user


@pytest_asyncio.fixture(scope="function")
async def test_custom_machine(app, paid_user):
    """Fixture that creates and tears down a test machine"""
    machine_data = {
        "name": "classic-custom-machine",
        "type": "classic",
        "endpoint": "http://localhost:8188",
        "auth_token": "test_auth_token",
    }

    async with get_test_client(app, paid_user) as client:
        response = await client.post("/machine/custom", json=machine_data)
        assert response.status_code == 200
        machine_id = response.json()["id"]
        print(f"Machine ID: {machine_id}")

        yield machine_id

        # Cleanup after test is done
        delete_response = await client.delete(f"/machine/{machine_id}")
        assert delete_response.status_code == 200
