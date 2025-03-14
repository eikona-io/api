import asyncio
import contextlib
from datetime import datetime, timedelta
from http.client import HTTPException
import logging
import os
from typing import Optional
from uuid import UUID, uuid4
from modal import Image, App, Secret, Sandbox, enable_output
import modal


app = App("session_manager")

current_directory = os.path.dirname(os.path.realpath(__file__))

image = (
    Image.debian_slim()
    .pip_install("aiohttp>=3.8.6", "python-jose>=3.3.0")
    .add_local_file(
        current_directory + "/extra_model_paths.yaml",
        "/extra_model_paths.yaml",
    )
)



ALGORITHM = "HS256"

os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2024.04"

async def send_log_async(
    update_endpoint: str, session_id: UUID, machine_id: str, log_message: str
):
    import aiohttp

    async with aiohttp.ClientSession() as client:
        token = generate_temporary_token("modal")
        async with client.post(
            update_endpoint + "/api/session/callback/log",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "session_id": str(session_id),
                "machine_id": machine_id,
                "log": log_message,
            },
        ) as response:
            pass
            # print("send_log_async", await response.text())


def send_log_entry(
    update_endpoint: str, session_id: UUID, machine_id: str, log_message: str
):  # noqa: F821
    asyncio.create_task(
        send_log_async(update_endpoint, session_id, machine_id, log_message)
    )
    # data = [
    #     (
    #         uuid4(),  # noqa: F821
    #         session_id,
    #         None,
    #         machine_id,
    #         datetime.now(),
    #         log_type,
    #         log_message,
    #     )
    # ]
    # asyncio.create_task(insert_to_clickhouse("log_entries", data))


logger = logging.getLogger(__name__)


# Secret.from_name("civitai-api-key")
@app.function(
    timeout=3600,
    secrets=[
        Secret.from_name("api_auth"),
        # Secret.from_name("clickhouse_prod"),
        # Secret.from_name("upstash_redis_prod"),
    ],
    image=image,
)
async def run_session(
    user_id: str,
    org_id: Optional[str],
    session_id: UUID,
    machine_id: str,
    machine_version_id: str,
    docker_config,
    gpu,
    timeout,
    workdir,
    encrypted_ports,
    update_endpoint,
    comfyui_cmd,
    shared_model_volume_name,
    tunnel_url_queue,
):
    os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2024.04"
    from modal._output import OutputManager

    class CustomOutputManager(OutputManager):
        _context_id = None
        _machine_id = None

        @classmethod
        @contextlib.contextmanager
        def enable_output_with_context(
            cls, context_id: str, machine_id: str, show_progress: bool = True
        ):
            print(f"[Intercepted Modal Log] enable_output with context {context_id}")
            if show_progress:
                cls._instance = CustomOutputManager()
                cls._context_id = context_id
                cls._machine_id = machine_id
            try:
                yield
            finally:
                cls._instance = None
                cls._context_id = None
                cls._machine_id = None

        def _print_log(self, fd: int, data: str) -> None:
            # context_info = f"[Context: {self._context_id} {self._machine_id}] " if self._context_id else ""
            # print(f"[Intercepted Modal Log] {context_info}_print_log", fd, data)

            if self._context_id is not None:
                send_log_entry(
                    update_endpoint, self._context_id, self._machine_id, data
                )
                # item = [
                #     (
                #         uuid4(),
                #         self._context_id,
                #         None,
                #         self._machine_id,
                #         datetime.now(),
                #         "info",
                #         data,
                #     )
                # ]
                # # print("inserting to clickhouse")
                # asyncio.create_task(insert_to_clickhouse("log_entries", item))

            super()._print_log(fd, data)

    modal._output.OutputManager = CustomOutputManager

    import aiohttp

    dockerfile_image: modal.Image = None

    volumes = {}
    # Mount shared models only if shared_model_volume_name is present
    if shared_model_volume_name:
        volumes["/public_models"] = modal.Volume.from_name(
            shared_model_volume_name,
            create_if_missing=True,
        )

    volumes["/private_models"] = modal.Volume.from_name(
        "models_" + (org_id if org_id is not None else user_id),
        create_if_missing=True,
    )
    
    modal_image_id = "modal_image_id" in docker_config and docker_config["modal_image_id"]
    
    if (machine_id is not None and modal_image_id is None):
        # Lets try to fetch the modal image id from the machine
        try:
            get_image_id = modal.Function.from_name(machine_id, "get_image_id")
            modal_image_id = await get_image_id.remote.aio()
            print(f"Fetched modal image id: {modal_image_id}")
        except Exception as e:
            logger.error(f"Error fetching modal image id: {str(e)}")

    if modal_image_id:
        logger.info(f"Using existing modal image {modal_image_id}")
        send_log_entry(
            update_endpoint,
            session_id,
            machine_id,
            f"Using existing modal image {modal_image_id}",
        )
        dockerfile_image = modal.Image.from_id(modal_image_id)
    else:
        # Python version and base docker image setup
        python_version = docker_config.get("python_version", "3.11")
        base_docker_image = docker_config.get("base_docker_image")

        if base_docker_image:
            dockerfile_image = modal.Image.from_registry(
                base_docker_image, add_python=python_version
            )
        else:
            dockerfile_image = modal.Image.debian_slim(python_version=python_version)

        install_custom_node_with_gpu = docker_config.get(
            "install_custom_node_with_gpu", False
        )

        # Apply docker commands if available
        docker_commands = docker_config.get("docker_commands")
        if docker_commands:
            for commands in docker_commands:
                dockerfile_image = dockerfile_image.dockerfile_commands(
                    commands,
                    gpu=gpu if install_custom_node_with_gpu else None,
                )

    # Always add these commands regardless of the path taken
    dockerfile_image = dockerfile_image.run_commands(
        [
            "rm -rf /private_models /comfyui/models /public_models",
            "ln -s /private_models /comfyui/models",
        ],
    )

    # Always add extra_model_paths.yaml
    # current_directory = os.path.dirname(os.path.realpath(__file__))
    dockerfile_image = dockerfile_image.add_local_file(
        "/extra_model_paths.yaml",
        "/comfyui/extra_model_paths.yaml",
    )

    if not dockerfile_image:
        raise HTTPException(
            status_code=400, detail="No dependencies or modal image id provided"
        )

    async def check_for_timeout(session_id: str):
        try:
            while True:
                async with aiohttp.ClientSession() as client:
                    token = generate_temporary_token(user_id, org_id)
                    async with client.post(
                        update_endpoint + "/api/session/callback/check-timeout",
                        headers={"Authorization": f"Bearer {token}"},
                        json={
                            "session_id": str(session_id),
                        },
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("continue", True):
                                await asyncio.sleep(1)
                            else:
                                send_log_entry(
                                    update_endpoint,
                                    session_id,
                                    machine_id,
                                    "Session closed due to timeout",
                                )
                                break
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.info(f"Timeout checker cancelled for session {session_id}")
            raise
        except Exception as e:
            logger.error(f"Error in timeout checker for session {session_id}: {str(e)}")
            raise

    task = asyncio.create_task(check_for_timeout(session_id))

    try:
        send_log_entry(
            update_endpoint,
            session_id,
            machine_id,
            f"Queuing {gpu} Container..." if gpu else "Queuing CPU Container...",
        )
        with CustomOutputManager.enable_output_with_context(
            str(session_id), machine_id
        ):
            with enable_output():
                send_log_entry(
                    update_endpoint, session_id, machine_id, "Creating Sandbox..."
                )
                sb = await Sandbox.create.aio(
                    image=dockerfile_image,
                    timeout=timeout,
                    gpu=gpu,
                    app=app,
                    workdir=workdir,
                    encrypted_ports=encrypted_ports,
                    volumes=volumes,
                )

                send_log_entry(
                    update_endpoint, session_id, machine_id, "Setting up tunnels..."
                )
                tunnels = await sb.tunnels.aio()
                tunnel = tunnels[8188]  # Access the tunnel after awaiting
                send_log_entry(
                    update_endpoint, session_id, machine_id, "Tunnel connected"
                )

                if tunnel_url_queue is not None:
                    await tunnel_url_queue.put.aio(tunnel.url)

                token = generate_temporary_token(user_id, org_id)
                async with aiohttp.ClientSession() as client:
                    async with client.post(
                        update_endpoint + "/api/session/callback",
                        headers={"Authorization": f"Bearer {token}"},
                        json={
                            "session_id": str(session_id),
                            "machine_version_id": machine_version_id,
                            "tunnel_url": tunnel.url,
                            "sandbox_id": sb.object_id,
                        },
                    ) as response:
                        print(await response.text())

                send_log_entry(
                    update_endpoint, session_id, machine_id, "Starting ComfyUI..."
                )

                # models_path_command = "rm -rf /private_models /comfyui/models /public_models && ln -s /private_models /comfyui/models"

                p = await sb.exec.aio(
                    "bash",
                    "-c",
                    comfyui_cmd,
                )

                async def log_stream(stream, stream_type: str):
                    try:
                        async for line in stream:
                            try:
                                # Add debug logging to see what we're receiving
                                if isinstance(line, bytes):
                                    logger.debug(f"Received bytes: {repr(line)}")

                                    # Handle decoding in a separate try block
                                    try:
                                        line = line.decode("utf-8", errors="replace")
                                    except UnicodeDecodeError as decode_err:
                                        logger.error(f"Decode error: {decode_err}")
                                        continue  # Skip this line and continue with the next one

                                    # Skip progress bar lines
                                    if any(
                                        char in line
                                        for char in [
                                            "█",
                                            "▮",
                                            "▯",
                                            "▏",
                                            "▎",
                                            "▍",
                                            "▌",
                                            "▋",
                                            "▊",
                                            "▉",
                                            "\r",
                                        ]
                                    ):
                                        continue

                                print(line, end="")
                                # data = [
                                #     (
                                #         uuid4(),
                                #         session_id,
                                #         None,
                                #         machine_id,
                                #         datetime.now(),
                                #         stream_type,
                                #         line,
                                #     )
                                # ]
                                # asyncio.create_task(
                                #     insert_to_clickhouse("log_entries", data)
                                # )
                                send_log_entry(
                                    update_endpoint, session_id, machine_id, line
                                )
                            except Exception as e:
                                logger.error(
                                    f"Inner error processing log line: {str(e)}"
                                )
                                continue  # Ensure we continue processing next lines
                    except Exception as e:
                        logger.error(f"Outer error in log stream: {str(e)}")

                # Re-raise if this is a critical error that should stop the stream
                # raise

                # Create tasks for both stdout and stderr
                stdout_task = asyncio.create_task(log_stream(p.stdout, "info"))
                stderr_task = asyncio.create_task(log_stream(p.stderr, "info"))

                # Wait for both streams to complete
                await asyncio.gather(stdout_task, stderr_task)

            await sb.wait.aio()
    except asyncio.CancelledError:
        print("Input cancellation")
        # try:
        #     if sb is not None:
        #         await sb.terminate.aio()
        # except Exception as e:
        #     logger.error(f"Unable to force close sandbox: {str(e)}")
        send_log_entry(update_endpoint, session_id, machine_id, "Input cancellation")
        async with aiohttp.ClientSession() as client:
            token = generate_temporary_token(user_id, org_id)
            async with client.delete(
                update_endpoint + "/api/session/" + str(session_id),
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                print(await response.text())
        if (sb is not None):
            await sb.terminate.aio()
    except Exception as e:
        # # This function is cancelled
        # if isinstance(e, asyncio.CancelledError):
        #     logger.info("Task was cancelled")
        #     try:
        #         if sb is not None:
        #             await sb.terminate.aio()
        #     except Exception as e:
        #         logger.error(f"Unable to force close sandbox: {str(e)}")
        
        # logger.error(f"Error in session: {str(e)}")
        
        send_log_entry(update_endpoint, session_id, machine_id, str(e))
        async with aiohttp.ClientSession() as client:
            token = generate_temporary_token(user_id, org_id)
            async with client.delete(
                update_endpoint + "/api/session/" + str(session_id),
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                print(await response.text())
        if tunnel_url_queue is not None:
            await tunnel_url_queue.put.aio({"error": str(e)})
        # raise
    finally:
        task.cancel()


def generate_temporary_token(
    user_id: str, org_id: Optional[str] = None, expires_in: str = "1h"
) -> str:
    from jose import jwt
    JWT_SECRET = os.getenv("JWT_SECRET")

    """
    Generate a temporary JWT token for the given user_id and org_id.

    Args:
        user_id (str): The user ID to include in the token.
        org_id (Optional[str]): The organization ID to include in the token, if any.
        expires_in (str): The expiration time for the token. Default is "1h".

    Returns:
        str: The generated JWT token.
    """
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=1),  # Default expiration of 1 hour
    }
    
    payload["modal"] = True

    if org_id:
        payload["org_id"] = org_id

    if expires_in != "1h":
        # Parse the expiration time
        value = int(expires_in[:-1])
        unit = expires_in[-1].lower()
        if unit == "m":
            delta = timedelta(minutes=value)
        elif unit == "h":
            delta = timedelta(hours=value)
        elif unit == "d":
            delta = timedelta(days=value)
        elif unit == "w":
            delta = timedelta(weeks=value)
        else:
            raise ValueError("Invalid expiration format. Use m, h, d, or w.")

        payload["exp"] = datetime.utcnow() + delta

    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)
