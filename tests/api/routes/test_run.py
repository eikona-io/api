import requests
import os


def get_ngrok_url():
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        return response.json()["tunnels"][0]["public_url"]
    except:
        return "http://localhost:8000"  # fallback url


def get_ngrok_url_with_retry(max_retries=5, delay=1):
    """Get ngrok URL with retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get("http://host.docker.internal:4040/api/tunnels")
            return response.json()["tunnels"][0]["public_url"]
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                return "http://localhost:8000"  # fallback url
            print(
                f"Failed to get ngrok URL (attempt {attempt + 1}/{max_retries}). Retrying..."
            )
            import time

            time.sleep(delay)


ngrok_url = get_ngrok_url_with_retry()

os.environ["CURRENT_API_URL"] = ngrok_url

from api.models import User  # noqa: E402
from api.routes.utils import generate_persistent_token  # noqa: E402
import pytest  # noqa: E402
from httpx import AsyncClient  # noqa: E402
import pytest_asyncio  # noqa: E402
from uuid import uuid4  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import AsyncAdaptedQueuePool  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

load_dotenv()

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


# logger = logging.getLogger(__name__)
# logger.info("hi there")
# logger.info(f"ngrok_url: {ngrok_url}")


@pytest.mark.asyncio
async def test_ngrok_setup(app):
    """Test to verify ngrok setup and accessibility"""
    print("\nhi there")
    print(f"ngrok_url: {ngrok_url}")
    assert ngrok_url is not None

    # Try to access the URL
    try:
        response = requests.get(ngrok_url)
        assert response.status_code == 200, (
            f"Expected status code 200, but got {response.status_code}"
        )
        print(
            f"Successfully connected to ngrok URL with status code {response.status_code}"
        )
    except requests.RequestException as e:
        pytest.fail(f"Failed to connect to ngrok URL: {str(e)}")


@pytest_asyncio.fixture(scope="session")
async def test_user():
    async with get_db_context() as db:
        user = User(
            id=str(uuid4()),
            username="test_user",
            name="Test User",
        )
        db.add(user)
    yield user
    # Skip deleting the user, beucase other data might depends on this
    # async with get_db_context() as db:
    #     await db.delete(user)
    #     await db.commit()


@pytest_asyncio.fixture(scope="session")
async def app(test_user):
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


@pytest_asyncio.fixture(scope="function")
async def client(app, test_user):
    api_key = generate_persistent_token(test_user.id, None)

    # Create the client with increased timeout
    client = AsyncClient(
        base_url=app + "/api", 
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0  # Add timeout in seconds (default is 5 seconds)
    )
    yield client
    # Cleanup
    await client.aclose()


@pytest.mark.asyncio
async def test_get_run_not_found(client):
    """Test getting a non-existent run returns 404"""
    random_uuid = uuid4()
    response = await client.get(f"/run/{random_uuid}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"


@pytest_asyncio.fixture(scope="function")
async def test_custom_machine(client):
    """Fixture that creates and tears down a test machine"""
    machine_data = {
        "name": "test-machine",
        "type": "classic",
        "endpoint": "http://localhost:8188",
        "auth_token": "test_auth_token",
    }
    print("Creating test machine")
    response = await client.post("/machine/custom", json=machine_data)
    assert response.status_code == 200
    machine_id = response.json()["id"]

    print(f"Machine ID: {machine_id}")

    yield machine_id  # Provide the machine_id to the test

    # Cleanup after test completes
    delete_response = await client.delete(f"/machine/{machine_id}")
    assert delete_response.status_code == 200


@pytest.mark.asyncio
async def test_get_test_machine(client, test_custom_machine):
    """Test getting a test machine"""
    response = await client.get(f"/machine/{test_custom_machine}")
    assert response.status_code == 200
    assert response.json()["id"] == test_custom_machine


@pytest_asyncio.fixture(scope="function")
async def test_serverless_machine(client):
    """Fixture that creates and tears down a test serverless machine"""
    machine_data = {
        "name": "test-serverless-machine",
        "gpu": "CPU",
        "wait_for_build": True,
    }
    print("Creating test serverless machine")
    response = await client.post("/machine/serverless", json=machine_data)
    assert response.status_code == 200
    machine_id = response.json()["id"]

    yield machine_id  # Provide the machine_id to the test

    # Cleanup after test completes
    delete_response = await client.delete(f"/machine/{machine_id}?force=true")
    assert delete_response.status_code == 200

# @pytest.mark.asyncio
# async def test_get_serverless_machine(client, test_serverless_machine):
#     """Test getting a serverless machine"""
#     response = await client.get(f"/machine/{test_serverless_machine}")
#     assert response.status_code == 200
#     assert response.json()["id"] == test_serverless_machine
    
    
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

@pytest_asyncio.fixture(scope="function")
async def test_create_workflow_deployment(client, test_serverless_machine):
    """Test creating a workflow and deployment"""
    workflow_data = {
        "name": "test-workflow",
        "workflow_json": basic_workflow_json,
        "workflow_api": basic_workflow_api_json,
        "machine_id": test_serverless_machine,
    }
    response = await client.post("/workflow", json=workflow_data)
    assert response.status_code == 200, f"Workflow creation failed with response: {response.text}"
    workflow_id = response.json()["workflow_id"]
    
    # Get the workflow version id
    response = await client.get(f"/workflow/{workflow_id}/versions")
    assert response.status_code == 200, f"Getting workflow versions failed with response: {response.text}"
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
    
    
@pytest.mark.asyncio
async def test_run_deployment_sync(client, test_create_workflow_deployment):
    """Test running a deployment"""
    deployment_id = test_create_workflow_deployment
    response = await client.post("/run/deployment/sync", json={"deployment_id": deployment_id})
    assert response.status_code == 200
    run_id = response.json()[0]["run_id"]
    assert run_id is not None