from config import config
import modal
from time import time
from modal import (
    Image,
    Mount,
    App,
    web_endpoint,
    Stub,
    asgi_app,
    method,
    enter,
    exit,
    forward,
    Dict,
    Queue,
    functions,
)
from typing import Union, Optional, Dict, List, Any, Annotated, cast
import json
import urllib.request
import urllib.parse
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException, WebSocket
from fastapi import UploadFile, File, Form
from pathlib import Path
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from volume_setup import (
    volumes,
    INPUT_DIR,
    OUTPUT_DIR,
    PRIVATE_BASEMODEL_DIR_SYM,
    PUBLIC_BASEMODEL_DIR,
    public_model_volume,
    private_volume,
)
from datetime import datetime, timezone
import aiohttp
from aiohttp import TCPConnector
import os
import uuid
from enum import Enum
import subprocess
import asyncio
from pprint import pprint
import time
from collections import defaultdict, deque

current_directory = os.path.dirname(os.path.realpath(__file__))

deploy_test = config["deploy_test"] == "True"

print("Builder Version: 3")
print("Builder Deps: ", os.getenv("MODAL_IMAGE_BUILDER_VERSION"))
print("Modal Version: ", modal.__version__)

app = App(name=config["name"])

legacy_mode = config["legacy_mode"]
skip_static_assets = config["skip_static_assets"]

if os.getenv("MODAL_ENVIRONMENT") == "dev":
    machine_name_for_label = config["name"][:14]
else:
    machine_name_for_label = config["name"]

compile_with_gpu = config["install_custom_node_with_gpu"] == "True"
final_gpu_param = str(config["gpu"]) if config["gpu"] != "CPU" else None
gpu_param = final_gpu_param if compile_with_gpu else None

deps = config["deps"]
docker_commands = config["docker_commands"]

base_docker_image = config["base_docker_image"]
python_version = (
    config["python_version"]
    if config["python_version"] is not None and config["python_version"] != ""
    else "3.11"
)
prestart_command = config["prestart_command"]
global_extra_args = config["extra_args"]

print(base_docker_image, python_version, prestart_command, global_extra_args)

dockerfile_image = modal.Image.debian_slim(python_version=python_version)

if base_docker_image is not None and base_docker_image != "":
    dockerfile_image = modal.Image.from_registry(
        base_docker_image, add_python=python_version
    )

# dockerfile_image = (
#     dockerfile_image
#     .run_commands("python --version")
#     .apt_install("git", "wget", "curl")
#     .apt_install("libgl1-mesa-glx", "libglib2.0-0")
#     # For uv local path
#     .env({"VIRTUAL_ENV": "/usr/local/"})
#     .pip_install("boto3", "aioboto3")
# )

print("final_gpu_param: ", final_gpu_param)
print("docker_commands: ", docker_commands)

# Install all custom nodes
if docker_commands is not None:
    for commands in docker_commands:
        dockerfile_image = dockerfile_image.dockerfile_commands(
            commands,
            gpu=gpu_param,
        )

dockerfile_image = dockerfile_image.copy_local_file(
    f"{current_directory}/data/extra_model_paths.yaml", "/comfyui"
)


async def get_static_assets(
    get_object_info=True,
    get_filename_list_cache=True,
    get_extensions=True,
    upload_extensions=True,
):
    import aioboto3
    import asyncio

    # print(f"aioboto3 version: {aioboto3.__version__}")
    # print(f"asyncio version: {asyncio.__version__}")
    # print(f"boto3 version: {boto3.__version__}")

    import json
    import subprocess

    bucket_name = "comfydeploy-fe-js-assets"
    import io

    directory_path = "/comfyui/models"
    if os.path.exists(directory_path):
        directory_contents = os.listdir(directory_path)
        directory_path = "/comfyui/models/ipadapter"
        print(directory_contents)
        if os.path.exists(directory_path):
            directory_contents = os.listdir(directory_path)
            print(directory_contents)
    else:
        print(f"Directory {directory_path} does not exist.")

    # Read and write the main.py file
    main_py_path = "/comfyui/main.py"

    # Read the contents of main.py
    with open(main_py_path, "r") as file:
        main_py_contents = file.read()
        original_main_py_contents = main_py_contents

    # Write a simple modification to main.py
    with open(main_py_path, "w") as file:
        main_py_contents = main_py_contents.replace(
            "import folder_paths",
            """
import folder_paths

original_get_filename_list = folder_paths.get_filename_list

# Redefine get_filename_list
def get_filename_list(filename):
    # some nodes break when we try to replace the filename list so instead let them do their output
    if filename == "VHS_video_formats":
        return original_get_filename_list(filename)
    return ["__"+filename+"__"]

# Override the original get_filename_list with the new one
folder_paths.get_filename_list = get_filename_list
                                 """,
        )
        print("Updated main.py with an additional comment")
        file.write(main_py_contents)

    # print(main_py_contents)

    server_process = await asyncio.subprocess.create_subprocess_shell(
        comfyui_cmd(cpu=True if gpu_param is None else False) + " --disable-metadata",
        cwd="/comfyui",
    )

    ok = await check_server(
        f"http://{COMFY_HOST}",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    )

    print("ok: ", ok)

    async def fetch_from_comfy(url_path, is_json=True):
        full_url = f"http://{COMFY_HOST}/{url_path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(full_url) as response:
                # print("response: ", response)
                if response.status == 200:
                    if is_json:
                        res = await response.json()
                    else:
                        res = await response.text()
                    return res
                else:
                    print(f"Failed to retrieve {url_path}")
                    return None

    object_info = None

    print("get_object_info")

    if get_object_info:
        object_info = await fetch_from_comfy("object_info")

    # print("object_info: ", object_info)
    # if get_object_info:
    #     json_file_path = "/comfyui/object_info_temp.json"
    #     subprocess.run(
    #         ["python", "/comfyui/static_fe_object_info_script.py", json_file_path],
    #         check=True,
    #     )
    #     with open(json_file_path, "r") as json_file:
    #         object_info = json.load(json_file)

    filename_list_cache = None

    # this popultes our filename_list_cache
    if get_filename_list_cache:
        await fetch_from_comfy("object_info")
        filename_list_cache = await fetch_from_comfy(
            "comfyui-deploy/filename_list_cache"
        )

    ext_map = {}
    if get_extensions:
        extension_list = await fetch_from_comfy("extensions")

        async def fetch_extension(ext_path):
            ext_path = ext_path.lstrip("/")
            js_file = await fetch_from_comfy(ext_path, False)
            return ext_path, js_file

        # Fetch all extensions concurrently
        results = await asyncio.gather(
            *[fetch_extension(ext_path) for ext_path in extension_list]
        )

        # Populate ext_map with the results
        ext_map = dict(results)

    if upload_extensions:
        import re

        async def upload_to_s3(ext_map, bucket_name, parent_path=""):
            async with aioboto3.Session().client(
                "s3",
            ) as s3:
                tasks = []
                for ext_path, js_file in ext_map.items():
                    # Read the content of the JavaScript file
                    js_content = js_file

                    if js_file is None:
                        print(f"js_file is None for {ext_path}")
                        continue

                    import_pattern = (
                        r'(import.*from\s*["\'])(\.\.\/\.\.\/)([^"\']+)(["\'])'
                    )

                    js_content = re.sub(
                        import_pattern,
                        lambda m: f"{m.group(1)}../../{m.group(2)}{m.group(3)}{m.group(4)}",
                        js_content,
                    )

                    full_key_path = (
                        f"{parent_path}/{ext_path}" if parent_path else ext_path
                    )
                    tasks.append(
                        s3.put_object(
                            Bucket=bucket_name,
                            Key=full_key_path,
                            Body=io.BytesIO(js_content.encode()),
                            ContentType="application/javascript",
                        )
                    )
                # Add task to upload extension-list.json
                if extension_list is not None:
                    extension_list_key = (
                        f"{parent_path}/extension-list.json"
                        if parent_path
                        else "extension-list.json"
                    )
                    tasks.append(
                        s3.put_object(
                            Bucket=bucket_name,
                            Key=extension_list_key,
                            Body=json.dumps(extension_list).encode(),
                            ContentType="application/json",
                        )
                    )

                # Add task to upload object_info.json
                if object_info is not None:
                    object_info_key = (
                        f"{parent_path}/object_info.json"
                        if parent_path
                        else "object_info.json"
                    )
                    tasks.append(
                        s3.put_object(
                            Bucket=bucket_name,
                            Key=object_info_key,
                            Body=json.dumps(object_info).encode(),
                            ContentType="application/json",
                        )
                    )

                print("tasks: Uploading")
                await asyncio.gather(*tasks)
                print("tasks: Uploaded")

        await upload_to_s3(ext_map, bucket_name, config["machine_id"])

    # Reset the file
    with open(main_py_path, "w") as file:
        file.write(original_main_py_contents)

    server_process.terminate()

    # return {
    #     "object_info": json.dumps(object_info),
    #     "filename_list_cache": filename_list_cache,
    #     "extensions": extension_list,
    # }


if skip_static_assets != "True":
    dockerfile_image = (
        dockerfile_image.workdir("/").run_function(
            get_static_assets,
            secrets=[modal.Secret.from_name("aws-s3-js-assets-user")],
            gpu=gpu_param,
        )
        # .pip_install("pydantic==1.10.14")  # NOTE: modal needs this pydantic version, custom nodes might install another version
    )

if legacy_mode == "False":
    dockerfile_image = dockerfile_image.run_commands(
        [
            f"rm -rf {PRIVATE_BASEMODEL_DIR_SYM}",
            "rm -rf /comfyui/models",
            f"ln -s {PRIVATE_BASEMODEL_DIR_SYM} /comfyui/models",
            f"rm -rf {PUBLIC_BASEMODEL_DIR}",
        ]
    )

# Start and check comfyui
# Skip checking comfyui

# dockerfile_image = (
#     dockerfile_image.workdir("/")
#     .copy_local_file(f"{current_directory}/data/install_models.py", "/")
#     # Will only start and check comfyui version
#     .run_commands(
#         f"python install_models.py{'' if compile_with_gpu else ' --cpu'}" ,
#         gpu=gpu_param
#     )
# )

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 1000
# Time to wait between poll attempts in milliseconds
COMFY_POLLING_INTERVAL_MS = 250
# Maximum number of poll attempts
COMFY_POLLING_MAX_RETRIES = 1000
# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"

input_directory = "/private_models/input"
output_directory = "/private_models/output"
temp_directory = "/private_models"


def comfyui_cmd(
    cpu: bool = False,
    extra_args: Optional[str] = None,
    mountIO: bool = False,
    dontCreateFolders: bool = False,
):
    if not dontCreateFolders:
        if not os.path.exists(input_directory):
            os.makedirs(input_directory)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)

    cmd = f"python main.py --dont-print-server --enable-cors-header --listen --port 8188 --input-directory {input_directory}"
    if mountIO:
        cmd += (
            f" --output-directory {output_directory} --temp-directory {temp_directory}"
        )
    if cpu:
        cmd += " --cpu"
    if extra_args is not None:
        cmd += f" {extra_args}"

    if global_extra_args is not None and global_extra_args != "":
        cmd += f" {global_extra_args}"

    if prestart_command is not None and prestart_command != "":
        cmd = f"{prestart_command} && {cmd}"

    print("Actual file command: ", cmd)

    return cmd


async def interrupt_comfyui():
    print("Interrupting comfyui")
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{COMFY_HOST}/interrupt") as response:
            # If the response status code is 200, the server is up and running
            if response.status == 200:
                return True
            else:
                return False


async def check_server(url, retries=50, delay=500):
    import aiohttp

    # for i in range(retries):
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    # If the response status code is 200, the server is up and running
                    if response.status == 200:
                        print(f"comfy-modal - API is reachable")
                        return True
        except Exception as e:
            # If an exception occurs, the server may not be ready
            pass

        # Wait for the specified delay before retrying
        await asyncio.sleep(delay / 1000)

    print(
        f"comfy-modal - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


async def check_server_with_log(
    url, retries=50, delay=500, machine_logs=[], last_sent_log_index=-1
):
    import aiohttp

    # for i in range(retries):
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    # If the response status code is 200, the server is up and running
                    if response.status == 200:
                        print(f"comfy-modal - API is reachable")
                        yield f"event: event_update\ndata: {json.dumps({'event': 'comfyui_api_ready'})}\n\n"
                        return

        except Exception as e:
            # If an exception occurs, the server may not be ready
            pass

        if machine_logs and last_sent_log_index != -1:
            while last_sent_log_index < len(machine_logs):
                log = machine_logs[last_sent_log_index]
                yield f"event: log_update\ndata: {json.dumps(log)}\n\n"
                last_sent_log_index += 1

        # Wait for the specified delay before retrying
        await asyncio.sleep(delay / 1000)


async def check_server_with_log_2(
    url, retries=50, delay=500, machine_logs=[], last_sent_log_index=-1
):
    import aiohttp

    # for i in range(retries):
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    # If the response status code is 200, the server is up and running
                    if response.status == 200:
                        print(f"comfy-modal - API is reachable")
                        # yield f"event: event_update\ndata: {json.dumps({'event': 'comfyui_api_ready'})}\n\n"
                        return

        except Exception as e:
            # If an exception occurs, the server may not be ready
            pass

        if machine_logs and last_sent_log_index != -1:
            new_logs = machine_logs[last_sent_log_index:]
            for log in new_logs:
                yield log
            last_sent_log_index = len(machine_logs)

        # Wait for the specified delay before retrying
        await asyncio.sleep(delay / 1000)


async def check_status(prompt_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://{COMFY_HOST}/comfyui-deploy/check-status?prompt_id={prompt_id}"
        ) as response:
            return await response.json()


async def check_ws_status(client_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://{COMFY_HOST}/comfyui-deploy/check-ws-status?client_id={client_id}"
        ) as response:
            return await response.json()


class Input(BaseModel):
    prompt_id: str
    workflow_api: Optional[dict] = None
    inputs: Optional[dict]
    workflow_api_raw: dict
    status_endpoint: str
    file_upload_endpoint: str
    workflow: Optional[dict] = None


async def queue_workflow_comfy_deploy(data: Input):
    data_str = data.json()
    data_bytes = data_str.encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['auth_token']}",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{COMFY_HOST}/comfyui-deploy/run", data=data_bytes, headers=headers
        ) as response:
            return await response.json()


class GPUEventType(str, Enum):
    GPU_START = "gpu_start"
    GPU_END = "gpu_end"


async def sync_report_gpu_event(
    event_id: str | None,
    is_workspace: bool = False,
    gpu: str | None = None,
    user_id: str | None = None,
    org_id: str | None = None,
    session_id: str | None = None,
) -> str:
    import requests
    from pprint import pprint

    event_type = GPUEventType.GPU_START if event_id is None else GPUEventType.GPU_END

    machine_id = config["machine_id"]
    auth_token = config["auth_token"]
    gpu_type = config["gpu"]  # Assuming this is properly set in your config
    gpu_event_callback_url = config["gpu_event_callback_url"]

    # Prepare the headers
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "bypass-tunnel-reminder": "true",
    }

    # Prepare the body with the current timestamp and optional instance_id
    body = {
        "machine_id": machine_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpuType": gpu_type if gpu is None else gpu,
        "eventType": event_type.value,
        "gpu_provider": "modal",
        "event_id": event_id,
        "is_workspace": is_workspace,
        "user_id": user_id,
        "org_id": org_id,
        "session_id": session_id,
        "modal_function_id": modal.current_function_call_id(),
    }

    print("body", body, gpu_event_callback_url)

    # Perform the asynchronous POST request
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                gpu_event_callback_url,
                json=body,
                headers=headers,
            ) as response:
                print(f"finished sending gpu event {event_type.value}")

                # Verify the response status
                if response.status != 200:
                    # Handle the error case appropriately; logging, raising an exception, etc.
                    print(f"Error: {response.status}")
                    response_data = (
                        await response.json()
                    )  # Adjust based on API response behavior
                    pprint(response_data)
                    return None
                else:
                    # Optionally process the response data
                    response_data = (
                        await response.json()
                    )  # Adjust based on API response behavior
                    pprint("SUCCESS")
                    pprint(response_data)
                    return response_data["event_id"]
        except aiohttp.ClientError as e:
            print(f"An error occurred during the request: {e}")
            return None


class GPUType(str, Enum):
    CPU = "CPU"
    T4 = "T4"
    A10G = "A10G"
    A100 = "A100"
    A100_80GB = "A100-80GB"
    H100 = "H100"
    L4 = "L4"


class OverrideInput(BaseModel):
    gpu: Optional[GPUType] = Field(default=None)
    cpu_only: Optional[bool] = Field(default=False)
    concurrency_limit: Optional[int] = Field(default=None)
    allow_concurrent_inputs: Optional[int] = Field(default=None)
    timeout: Optional[int] = Field(default=None)
    container_timeout: Optional[int] = Field(default=None)
    container_idle_timeout: Optional[int] = Field(default=None)
    private_volume_name: Optional[str] = Field(default=None)
    mountIO: Optional[bool] = Field(default=False)
    is_workspace: Optional[bool] = Field(default=False)
    user_id: Optional[str] = Field(default=None)
    org_id: Optional[str] = Field(default=None)
    kill: Optional[bool] = Field(default=False)
    extend_timeout: Optional[int] = Field(default=None)


class RequestInput(OverrideInput):
    input: Input


image = Image.debian_slim()

target_image = image if deploy_test else dockerfile_image

run_timeout = config["run_timeout"]
idle_timeout = config["idle_timeout"]
ws_timeout = config["ws_timeout"]


modal_secrets = []
if "TAILSCALE_AUTHKEY" in config and config["TAILSCALE_AUTHKEY"] is not None:
    modal_secrets = [
        modal.Secret.from_dict(
            {
                "ALL_PROXY": "socks5://localhost:1080/",
                "HTTP_PROXY": "http://localhost:1080/",
                "http_proxy": "http://localhost:1080/",
            }
        ),
        modal.Secret.from_dict(
            {
                "TAILSCALE_AUTHKEY": config["TAILSCALE_AUTHKEY"],
            }
        ),
    ]


async def send_status_update(input: Input, status: str, function_id: str = None):
    print("sending status update", input.status_endpoint, status, function_id)
    async with aiohttp.ClientSession() as session:
        data = {
            "run_id": input.prompt_id,
            "status": status,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        if function_id is not None:
            data["modal_function_call_id"] = function_id
        async with session.post(
            input.status_endpoint,
            data=json.dumps(data).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "bypass-tunnel-reminder": "true",
                "Authorization": f"Bearer {config['auth_token']}",
            },
        ) as response:
            pass


async def send_log_update(input: Input, log: str):
    async with aiohttp.ClientSession() as session:
        data = {
            "run_id": input.prompt_id,
            "log": log,
        }
        async with session.post(
            input.status_endpoint,
            data=json.dumps(data).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "bypass-tunnel-reminder": "true",
                "Authorization": f"Bearer {config['auth_token']}",
            },
        ) as response:
            pass


def create_web_app():
    web_app = FastAPI()

    def get_model(request_inputs: OverrideInput):
        # if request_inputs.gpu is not None or request_inputs.concurrency_limit is not None or request_inputs.private_volume_name is not None or request_inputs.timeout is not None:

        if request_inputs.private_volume_name is not None:
            _volumes = {
                PUBLIC_BASEMODEL_DIR: modal.Volume.lookup(
                    config["public_model_volume"], create_if_missing=True
                ),
                PRIVATE_BASEMODEL_DIR_SYM: modal.Volume.lookup(
                    request_inputs.private_volume_name, create_if_missing=True
                ),
            }
        else:
            _volumes = volumes

        print(config["public_model_volume"], request_inputs.private_volume_name)
        print(_volumes)
        # print(request_inputs.json())

        try:
            Model = modal.Cls.lookup(config["name"], "ComfyDeployRunner")
            if request_inputs.cpu_only:
                Model2 = Model.with_options(
                    gpu=None,
                    concurrency_limit=config["concurrency_limit"],
                )
                return Model2

            Model2 = Model.with_options(
                gpu=request_inputs.gpu.value
                if request_inputs.gpu is not None
                else None,
                concurrency_limit=request_inputs.concurrency_limit
                if request_inputs.concurrency_limit is not None
                else None,
                allow_concurrent_inputs=request_inputs.allow_concurrent_inputs
                if request_inputs.allow_concurrent_inputs is not None
                else 1,
                # We match the idle timeout
                container_idle_timeout=request_inputs.container_idle_timeout
                if request_inputs.container_idle_timeout is not None
                else None,
                volumes=_volumes,
                timeout=request_inputs.container_timeout
                if request_inputs.container_timeout is not None
                else None,
                # allow_background_volume_commits=True,
            )
            return Model2
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error spawning function"
            ) from e
        # else:
        #     return ComfyDeployRunner

    async def post_run(request: Request):
        try:
            # Parse the request body to a dict
            request_dict = await request.json()
            # print(request_dict)
            # Create a Pydantic model instance from the request dict
            request_input = RequestInput(**request_dict)
        except Exception as e:
            print(e)
            # Handle parsing/validation errors and return the exception details
            raise HTTPException(
                status_code=422, detail=f"Invalid request body: {str(e)}"
            ) from e

        # result = get_model(request_input)
        # result = result(
        #     volume_name=request_input.private_volume_name,
        #     mountIO=False,
        #     is_workspace=request_input.is_workspace,
        #     user_id=request_input.user_id,
        #     org_id=request_input.org_id,
        #     gpu=request_input.gpu,
        #     timeout=request_input.timeout
        # ).run.spawn(request_input.input)
        result = ComfyDeployRunner.run.spawn(request_input.input)

        # asyncio.create_task(send_status_update(request_input.input, "queued", result.object_id))
        asyncio.create_task(
            send_status_update(request_input.input, "not-started", result.object_id)
        )
        return JSONResponse({"call_id": result.object_id})

    async def post_run_streaming(request: Request):
        try:
            # Parse the request body to a dict
            request_dict = await request.json()
            # Create a Pydantic model instance from the request dict
            request_input = RequestInput(**request_dict)
            request_input.container_idle_timeout = 1200
            request_input.container_timeout = 3600
        except Exception as e:
            print(e)
            # Handle parsing/validation errors (e.g., return an HTTP 422 error)
            raise HTTPException(status_code=422, detail="Invalid request body") from e

        print(request_input.json())

        if request_input.kill:
            result = get_model(request_input)
            model = result(
                volume_name=request_input.private_volume_name,
                mountIO=False,
                is_workspace=request_input.is_workspace,
                user_id=request_input.user_id,
                org_id=request_input.org_id,
                gpu=request_input.gpu,
                timeout=request_input.timeout,
            )
            await model.streaming.spawn.aio(request_input.input, True)
            return {"message": "Run streaming killed"}

        if request_input.extend_timeout is not None:
            result = get_model(request_input)
            model = result(
                volume_name=request_input.private_volume_name,
                mountIO=False,
                is_workspace=request_input.is_workspace,
                user_id=request_input.user_id,
                org_id=request_input.org_id,
                gpu=request_input.gpu,
                timeout=request_input.timeout,
            )
            await model.streaming.spawn.aio(
                request_input.input, False, request_input.extend_timeout
            )
            return {"message": "Run streaming extended"}

        async def wrapped_generator():
            # Send initial padding to flush the buffer
            # print("Sending pad")
            # padding = ':' + ':' * 8192 + '\n'  # 2kB of padding
            # yield padding.encode("utf-8")
            warming_up = {"event": "warming_gpu"}
            yield f"event: event_update\ndata: {json.dumps(warming_up)}\n\n".encode(
                "utf-8"
            )

            # Ensure the client processes this immediately if possible
            # await asyncio.sleep(0)  # This can help in yielding control back to the event loop
            result = get_model(request_input)
            model = result(
                volume_name=request_input.private_volume_name,
                mountIO=False,
                is_workspace=request_input.is_workspace,
                user_id=request_input.user_id,
                org_id=request_input.org_id,
                gpu=request_input.gpu,
                timeout=request_input.timeout,
            )

            asyncio.create_task(
                send_status_update(
                    request_input.input, "not-started", modal.current_function_call_id()
                )
            )
            # asyncio.create_task(send_status_update(request_input.input, "queued", modal.current_function_call_id()))

            async for event in model.streaming.remote_gen.aio(request_input.input):
                yield event

        return StreamingResponse(
            wrapped_generator(),
            media_type="text/event-stream",
        )

    async def ws(websocket: WebSocket):
        await ComfyDeployRunner().websocket_endpoint.run(websocket)
        # await self.websocket_endpoint(websocket)

    async def view(
        volume_name: str,
        filename: str,
        type: str = "output",
        subfolder: str = None,
        gpu: str = None,
        timeout: int = None,
        user_id: str = None,
        org_id: str = None,
        from_volume: bool = False,
    ):
        # Security check to prevent path traversal
        if filename.startswith("/") or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = None

        # fix this
        if type == "output":
            file_path = "output"
        elif type == "input":
            file_path = "input"
        elif type == "temp":
            file_path = "temp"
        # fix this
        # elif type == "preview":
        # file_path = "output/preview"

        if subfolder is not None and len(subfolder) > 0:
            file_path += "/" + subfolder
        file_path += "/" + filename

        # Determine the content type based on the file extension
        content_type = "application/octet-stream"
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            content_type = f"image/{filename.split('.')[-1].lower()}"

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": content_type,
        }

        if from_volume:
            # read directly
            # Lookup the volume
            volume = modal.Volume.lookup(volume_name, create_if_missing=True)

            # Read the file from the volume
            async def file_streamer():
                retries = 10
                while retries > 0:
                    try:
                        async for chunk in volume.read_file.aio(file_path):
                            yield chunk
                        return  # Exit if successful
                    except Exception as e:
                        print(f"Error reading file: {e}")
                        retries -= 1
                        if retries == 0:
                            raise HTTPException(
                                status_code=404, detail="File not found"
                            )
                        await asyncio.sleep(1)  # Wait for 1 second before retrying

            return StreamingResponse(file_streamer(), headers=headers)
        else:
            file_path = "/comfyui/" + file_path
            if type == "input":
                file_path = file_path.replace("/comfyui", "/private_models")

            print("file_path", file_path)

        # if (type == "output"):
        request_input = OverrideInput(
            concurrency_limit=1,
            container_idle_timeout=1200,
            allow_concurrent_inputs=100,
            container_timeout=3600,
            gpu=gpu,
            # timeout=request_input.timeout,
            private_volume_name=volume_name,
            timeout=timeout,
            user_id=user_id,
            org_id=org_id,
            # backwards compatibility
            # mountIO=True,
        )

        result = get_model(request_input)
        result = result(
            volume_name=request_input.private_volume_name,
            mountIO=False,
            is_workspace=request_input.is_workspace,
            user_id=request_input.user_id,
            org_id=request_input.org_id,
            gpu=request_input.gpu,
            timeout=request_input.timeout,
        )

        return StreamingResponse(
            result.read_output_file.remote_gen(file_path), headers=headers
        )

    async def upload(
        file: Annotated[UploadFile, Form()],
        volume_name: Annotated[str, Form()],
        filename: Annotated[str, Form()],
        target_path: Annotated[str, Form()] = "/input",
        subfolder: Annotated[str, Form()] = None,
    ):
        # Lookup the volume
        from tempfile import gettempdir

        # Create a unique file path to avoid conflicts and overwrites
        local_location = Path(gettempdir()) / f"{uuid.uuid4()}_{filename}"
        file_location = Path(target_path) / f"{filename}"
        if subfolder is not None:
            file_location = Path(target_path) / subfolder / f"{filename}"
        print(local_location, file_location, volume_name)

        volume = await modal.Volume.lookup.aio(volume_name, create_if_missing=True)

        try:
            # Write file to local storage
            with local_location.open("wb") as buffer:
                data = await file.read()
                buffer.write(data)

            # Upload file to the volume
            async with await volume.batch_upload.aio() as batch:
                batch.put_file(str(local_location), str(file_location))

            # Delete the local file after uploading
            local_location.unlink()

        except Exception as e:
            # Provide a more specific error message
            raise HTTPException(
                status_code=500, detail=f"Failed to upload file: {str(e)}"
            )

        return {"message": "File uploaded successfully", "path": str(file_location)}

    async def static_assets(
        private_volume_name: Optional[str] = None,
        get_object_info=True,
        get_filename_list_cache=True,
        get_extensions=True,
        upload_extensions=True,
    ):
        # request_input = OverrideInput()
        # request_input.cpu_only = True
        print("before getting model")
        # comfy_runner = get_model(request_input)
        print("after model")
        result = await get_static_assets.remote.aio(
            get_object_info, get_filename_list_cache, get_extensions, upload_extensions
        )
        print("static assets retrieved")
        # print(result)
        return result

    web_app.add_api_route("/run", post_run, methods=["POST"])
    web_app.add_api_route("/run/streaming", post_run_streaming, methods=["POST"])
    web_app.add_api_websocket_route("/comfyui-deploy/ws", ws)
    web_app.add_api_route("/view", view, methods=["GET"])
    web_app.add_api_route("/upload", upload, methods=["POST"])
    web_app.add_api_route("/static-assets", static_assets, methods=["GET"])

    origins = [
        "*",
    ]

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return web_app


web_app = create_web_app()


@app.function(
    timeout=600,
    allow_concurrent_inputs=100,
)
@asgi_app(label=machine_name_for_label + "-comfyui-api")
def comfyui_api():
    return web_app


@app.cls(
    image=target_image,
    gpu=final_gpu_param,
    volumes=volumes,
    timeout=(config["run_timeout"] + 20),
    container_idle_timeout=config["idle_timeout"],
    allow_concurrent_inputs=config["allow_concurrent_inputs"],
    concurrency_limit=config["concurrency_limit"],
    secrets=modal_secrets + [modal.Secret.from_name("aws-s3-js-assets-user")],
    # _allow_background_volume_commits=True,
)
class ComfyDeployRunner:
    web_app = FastAPI()
    machine_logs = []
    last_sent_log_index = 0

    cleanup_done = False
    current_function_call_id = None
    status_endpoint = None
    current_input = None

    async def timeout_and_exit(self, timeout_seconds: int, soft_exit: bool = False):
        import os

        await asyncio.sleep(timeout_seconds)
        print(f"Exiting container after {timeout_seconds} seconds.")
        # await asyncio.get_event_loop().shutdown_default_executor()  # Gracefully stop the event loop
        print(f"comfy-modal - cleanup")

        await sync_report_gpu_event(
            self.gpu_event_id, self.is_workspace, self.gpu, self.user_id, self.org_id
        )
        self.stdout_task.cancel()
        self.stderr_task.cancel()

        current_function_call_id = (
            self.current_function_call_id
        )  # modal.current_function_call_id()
        print("current_function_call_id", current_function_call_id)
        try:
            print("Interrupting comfyui")
            ok = await interrupt_comfyui()
        except Exception as e:
            print("Issues when interrupting comfyui", e)
            pass

        try:
            modal.functions.FunctionCall.from_id(current_function_call_id).cancel()
        except Exception as e:
            print("Issues when canceling function call", e)
            pass

        print(f"comfy-modal - cleanup done")
        if not soft_exit:
            os._exit(0)

    def __init__(
        self,
        volume_name: Optional[str] = None,
        mountIO: Optional[bool] = False,
        is_workspace: Optional[bool] = False,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        gpu: Optional[str] = None,
        timeout: Optional[int] = None,
        session_id: Optional[str] = False,
        workspace_tunnel: Optional[bool] = False,
    ) -> None:
        if volume_name is None:
            volume_name = config["private_model_volume"]
        self.private_volume = modal.Volume.lookup(volume_name, create_if_missing=True)
        self.mountIO = mountIO
        self.is_workspace = is_workspace
        self.user_id = user_id
        self.org_id = org_id
        self.gpu = gpu
        self.timeout = timeout
        self.cold_start_queue = deque()
        self.log_queues = deque()
        self.workspace_tunnel = workspace_tunnel
        self.session_id = session_id

    # self.log_task = None

    async def process_log_queue(self):
        try:
            while True:
                if len(self.cold_start_queue) > 0 and self.status_endpoint:
                    logs = list(self.cold_start_queue)
                    print("sending log batch", len(logs))
                    await self.send_log_batch(
                        run_id=self.current_input.prompt_id
                        if self.current_input
                        else None,
                        session_id=self.session_id,
                        logs=logs,
                        status_endpoint=self.status_endpoint,
                    )
                    self.cold_start_queue.clear()

                if self.log_queues and len(self.log_queues) > 0:
                    # Get the first item (prompt_id and queue_data) from the log_queues
                    logs = list(self.log_queues[0]["logs"])
                    input = self.log_queues[0]["current_input"]
                    if logs:
                        await self.send_log_batch(
                            run_id=input.prompt_id,
                            session_id=self.session_id,
                            logs=logs,
                            status_endpoint=input.status_endpoint,
                        )
                        self.log_queues.clear()

                await asyncio.sleep(1)
        except Exception as e:
            print("Error in process_log_queue", e)

    async def send_log_batch(self, run_id, session_id, logs, status_endpoint):
        if not logs:
            return  # Don't send empty log batches

        async with aiohttp.ClientSession() as session:
            data = json.dumps(
                {
                    "run_id": run_id,
                    "session_id": session_id,
                    "machine_id": config["name"],
                    "logs": logs,
                }
            ).encode("utf-8")
            print("sending log batch", run_id, session_id, status_endpoint)
            try:
                async with session.post(
                    status_endpoint,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "bypass-tunnel-reminder": "true",
                        "Authorization": f"Bearer {config['auth_token']}",
                    },
                ) as response:
                    if response.status != 200:
                        print(
                            f"Failed to send logs for run_id {run_id}. Status: {response.status}"
                        )
            except Exception as e:
                print(f"Error sending logs for run_id {run_id}: {str(e)}")

    @modal.method()
    async def streaming(
        self, input: Input, kill: bool = False, extend_timeout: Optional[int] = None
    ):
        try:
            print("post_run_streaming")

            self.current_function_call_id = modal.current_function_call_id()

            asyncio.create_task(send_status_update(input, "queued"))

            if extend_timeout is not None:
                remaining_time = self.timeout - (time.time() - self.timeout_start_time)
                self.timeout_task.cancel()
                self.timeout_task = asyncio.create_task(
                    self.timeout_and_exit(remaining_time + extend_timeout)
                )
                raise Exception("Extended Timeout")

            if kill:
                asyncio.create_task(self.timeout_and_exit(0))
                raise Exception("Killed")

            yield f"event: event_update\ndata: {json.dumps({'event': 'gpu_ready'})}\n\n"
            yield f"event: event_update\ndata: {json.dumps({'event': 'gpu_event_id', 'data': self.gpu_event_id})}\n\n"

            # ok = await check_server(
            #     f"http://{COMFY_HOST}",
            #     COMFY_API_AVAILABLE_MAX_RETRIES,
            #     COMFY_API_AVAILABLE_INTERVAL_MS,
            # )

            # self.private_volume.reload()

            async for event in check_server_with_log(
                f"http://{COMFY_HOST}",
                COMFY_API_AVAILABLE_MAX_RETRIES,
                COMFY_API_AVAILABLE_INTERVAL_MS,
                self.machine_logs,
                self.last_sent_log_index,
            ):
                yield event

            await send_status_update(input, "started")

            yield f"event: event_update\ndata: {json.dumps({'event': 'comfyui_ready'})}\n\n"

            # if not ok:
            #     raise Exception("ComfyUI API is not available")

            async with aiohttp.ClientSession() as session:
                data_str = input.json()
                data_bytes = data_str.encode("utf-8")
                async with session.post(
                    f"http://{COMFY_HOST}/comfyui-deploy/run/streaming",
                    data=data_bytes,
                    headers={"Authorization": f"Bearer {config['auth_token']}"},
                ) as response:
                    async for event in response.content.iter_any():
                        # print(event)
                        yield event

                        # Send the machine logs
                        while self.last_sent_log_index < len(self.machine_logs):
                            log = self.machine_logs[self.last_sent_log_index]
                            yield f"event: log_update\ndata: {json.dumps(log)}\n\n"
                            self.last_sent_log_index += 1

            # current_directories = os.listdir('/comfyui/output')
            # print("Finished Current directories in '/comfyui/output':", current_directories)

            # current_directories = os.listdir(output_directory)
            # print("Finished Current directories in :", output_directory, current_directories)
        except asyncio.CancelledError as e:
            print("This got fucking cancelled", e)  # this doesn't happen
            try:
                ok = await interrupt_comfyui()
            except Exception as e:
                pass
            finally:
                await send_status_update(input, "cancelled")

    @modal.method()
    async def read_output_file(self, file):
        # Read the file from the volume and yield its contents as bytes
        try:
            with open(file, "rb") as f:
                while True:
                    chunk = f.read()  # Read as much as possible
                    if not chunk:
                        break
                    yield chunk
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    @modal.method()
    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()

        query_params = websocket.query_params
        query_string = urllib.parse.urlencode(query_params)
        sid = uuid.uuid4().hex
        ws_url = (
            f"http://127.0.0.1:8188/comfyui-deploy/ws?{query_string}&clientId={sid}"
        )

        try:
            ok = await check_server(
                f"http://{COMFY_HOST}",
                COMFY_API_AVAILABLE_MAX_RETRIES,
                COMFY_API_AVAILABLE_INTERVAL_MS,
            )

            # ws_timeout = 2

            if not ok:
                raise Exception("ComfyUI API is not available")

            async def timeout_check(sid: str):
                wait_time = 0
                while True:
                    data = await check_ws_status(sid)
                    # print(data)
                    if "remaining_queue" in data and data["remaining_queue"] == 0:
                        if wait_time >= ws_timeout:
                            return True

                        await asyncio.sleep(0.10)
                        wait_time += 0.10
                    else:
                        await asyncio.sleep(0.10)

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url) as local_ws:

                        async def websocket_to_local():
                            try:
                                while True:
                                    ws_receive_task = asyncio.create_task(
                                        websocket.receive()
                                    )
                                    # Prepare the check_ws_status coroutine (with some delay if needed)
                                    check_status_task = asyncio.create_task(
                                        timeout_check(sid)
                                    )

                                    done, pending = await asyncio.wait(
                                        {ws_receive_task, check_status_task},
                                        return_when=asyncio.FIRST_COMPLETED,
                                    )

                                    for task in pending:
                                        task.cancel()

                                    if ws_receive_task in done:
                                        ws_msg = ws_receive_task.result()

                                        # Timeout for the input
                                        # ws_msg = await asyncio.wait_for(websocket.receive(), ws_timeout)
                                        # print(ws_msg)

                                        if (
                                            "type" in ws_msg
                                            and ws_msg["type"] == "websocket.disconnect"
                                        ):
                                            local_to_websocket_task.cancel()
                                            websocket_to_local_task.cancel()
                                            print("cancelled from remote client")
                                            break

                                        if (
                                            "bytes" in ws_msg
                                            and ws_msg["bytes"] is not None
                                        ):
                                            print("Received binary data")
                                            await local_ws.send_bytes(ws_msg["bytes"])
                                        elif (
                                            "text" in ws_msg
                                            and ws_msg["text"] is not None
                                        ):
                                            print("Received text data")
                                            await local_ws.send_str(ws_msg["text"])

                                    if (
                                        check_status_task in done
                                        and check_status_task.result()
                                    ):
                                        raise asyncio.TimeoutError(
                                            "Queue processing timeout."
                                        )

                            except asyncio.TimeoutError:
                                local_to_websocket_task.cancel()
                                websocket_to_local_task.cancel()
                                print(
                                    f"Timeout: No message received in {ws_timeout} seconds."
                                )
                                await websocket.close(
                                    1000,
                                    reason=f"Timeout: No message received in {ws_timeout} seconds.",
                                )
                            except asyncio.CancelledError:
                                await local_ws.close()

                        async def local_to_websocket():
                            try:
                                while True:
                                    msg = await local_ws.receive()  # await asyncio.wait_for(, ws_timeout)  # 10-second timeout
                                    # print(msg.type.name)
                                    if msg.type in [
                                        aiohttp.WSMsgType.CLOSE,
                                        aiohttp.WSMsgType.CLOSED,
                                    ]:
                                        local_to_websocket_task.cancel()
                                        websocket_to_local_task.cancel()
                                        print("cancelled from localhost")
                                        break
                                    elif msg.type in [aiohttp.WSMsgType.TEXT]:
                                        await websocket.send_text(msg.data)
                                    elif msg.type in [aiohttp.WSMsgType.BINARY]:
                                        await websocket.send_bytes(msg.data)
                            except asyncio.TimeoutError:
                                print(
                                    f"Timeout: No message received in {ws_timeout} seconds."
                                )
                                await websocket.close(
                                    1000,
                                    reason=f"Timeout: No message received in {ws_timeout} seconds.",
                                )
                            except asyncio.CancelledError:
                                pass  # Task was cancelled, ignore
                                await websocket.close()

                        websocket_to_local_task = asyncio.create_task(
                            websocket_to_local()
                        )
                        local_to_websocket_task = asyncio.create_task(
                            local_to_websocket()
                        )

                        try:
                            await asyncio.gather(
                                websocket_to_local_task, local_to_websocket_task
                            )
                        except asyncio.CancelledError:
                            pass  # Task was cancelled, ignore

                        print("both task finished")
            except Exception as e:
                print(f"Error in WebSocket communication: {e}")
            finally:
                pass
                # if websocket.state !=
                # await websocket.close()
                # print("closing for ws");

        except Exception as e:
            pass
        finally:
            pass
            # stdout_task.cancel()
            # stderr_task.cancel()
            # await stdout_task
            # await stderr_task

    async def read_stream(self, stream, isStderr):
        import time

        while True:
            try:
                line = await stream.readline()
                if line:
                    l = line.decode("utf-8").strip()

                    if l == "":
                        continue

                    if self.log_queues is not None and len(self.log_queues) > 0:
                        target_log = self.log_queues[0]["logs"]
                    else:
                        target_log = self.cold_start_queue

                    target_log = cast(deque, target_log)
                    target_log.append({"logs": l, "timestamp": time.time()})
                    # print("appending to log queue", len(target_log), target_log)

                    if not isStderr:
                        print(l, flush=True)
                        self.machine_logs.append({"logs": l, "timestamp": time.time()})
                    else:
                        print(l, flush=True)
                        self.machine_logs.append({"logs": l, "timestamp": time.time()})
                else:
                    break
            except asyncio.CancelledError:
                # Handle the cancellation here if needed
                break  # Break out of the loop on cancellation

    current_tunnel_url = ""

    @modal.method()
    async def create_tunnel(self, q, status_endpoint):
        self.status_endpoint = status_endpoint
        print("status_endpoint", status_endpoint)

        ok = await check_server(
            f"http://{COMFY_HOST}",
            COMFY_API_AVAILABLE_MAX_RETRIES,
            COMFY_API_AVAILABLE_INTERVAL_MS,
        )

        if self.current_tunnel_url != "":
            await q.put.aio(self.current_tunnel_url)
            return

        with modal.forward(8188) as tunnel:
            self.current_tunnel_url = tunnel.url
            self.current_function_call_id = modal.current_function_call_id()

            await q.put.aio(tunnel.url)

            print(f"tunnel.url        = {tunnel.url}")
            print(f"tunnel.tls_socket = {tunnel.tls_socket}")

            # Wait for the server process to exit
            try:
                while True:
                    await asyncio.sleep(1)  # Che  ck every 1 seconds
            except asyncio.CancelledError:
                print("cancelled")
                try:
                    ok = await interrupt_comfyui()
                except Exception as e:
                    pass

    @modal.method()
    async def close_container(self):
        await self.timeout_and_exit(0, True)

    @enter()
    async def setup(self):
        # Make sure that the ComfyUI API is available
        print(f"comfy-modal - check server")
        # reload volumes
        public_model_volume.reload()
        self.private_volume.reload()

        directory_path = "/comfyui/models"
        if os.path.exists(directory_path):
            directory_contents = os.listdir(directory_path)
            directory_path = "/comfyui/models/ipadapter"
            print(directory_contents)
            if os.path.exists(directory_path):
                directory_contents = os.listdir(directory_path)
                print(directory_contents)
        else:
            print(f"Directory {directory_path} does not exist.")

        self.server_process = await asyncio.subprocess.create_subprocess_shell(
            comfyui_cmd(mountIO=self.mountIO) + " --disable-metadata",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/comfyui",
            # env={**os.environ, "COLUMNS": "10000"}
            # env={
            #     "ALL_PROXY": "socks5://localhost:1080/",
            #     "HTTP_PROXY": "http://localhost:1080/",
            #     "http_proxy": "http            #     "TAILSCALE_AUTHKEY": config["TAILSCALE_AUTHKEY"],
            # },
        )

        print("setting up stdout and stderr")

        self.stdout_task = asyncio.create_task(
            self.read_stream(self.server_process.stdout, False)
        )
        self.stderr_task = asyncio.create_task(
            self.read_stream(self.server_process.stderr, True)
        )

        print("setting up log queue 2")

        self.log_task = asyncio.create_task(self.process_log_queue())

        print("setting up gpu event id")

        self.gpu_event_id = await sync_report_gpu_event(
            None,
            self.is_workspace,
            self.gpu,
            self.user_id,
            self.org_id,
            self.session_id,
        )

        print("setting up timeout")

        if self.timeout is not None:
            print("timeout", self.timeout)
            self.timeout_start_time = time.time()
            self.timeout_task = asyncio.create_task(self.timeout_and_exit(self.timeout))

    @exit()
    async def cleanup(self):
        if self.cleanup_done:
            return

        print(f"comfy-modal - cleanup")

        await sync_report_gpu_event(
            self.gpu_event_id, self.is_workspace, self.gpu, self.user_id, self.org_id
        )
        self.stdout_task.cancel()
        self.stderr_task.cancel()
        self.log_task.cancel()
        # await self.stdout_task
        # await self.stderr_task

        # try:
        #     self.private_volume.remove_file("temp", recursive=True)
        #     self.private_volume.remove_file("output", recursive=True)
        # except Exception as e:
        #     print("Issues when cleaning up", e)

        print(f"comfy-modal - cleanup done")

    @modal.method()
    async def run(self, input: Input):
        if isinstance(input, dict):
            input = Input(**input)

        try:
            self.log_queues.append({"logs": deque(), "current_input": input})

            if len(self.log_queues) == 1:
                self.current_input = input
                self.status_endpoint = input.status_endpoint

            import signal
            import time

            print("Got input")

            # self.private_volume.reload()

            await send_status_update(input, "queued", modal.current_function_call_id())

            result = {"status": "queued"}

            class TimeoutError(Exception):
                pass

            def timeout_handler(signum, frame):
                data = json.dumps(
                    {
                        "run_id": input.prompt_id,
                        "status": "timeout",
                        "time": datetime.now(timezone.utc).isoformat(),
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    input.status_endpoint,
                    data=data,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "bypass-tunnel-reminder": "true",
                        "Authorization": f"Bearer {config['auth_token']}",
                    },
                )
                urllib.request.urlopen(req)
                raise TimeoutError("Operation timed out")

            signal.signal(signal.SIGALRM, timeout_handler)

            try:
                signal.alarm(run_timeout)

                ok = await check_server(
                    f"http://{COMFY_HOST}",
                    COMFY_API_AVAILABLE_MAX_RETRIES,
                    COMFY_API_AVAILABLE_INTERVAL_MS,
                )

                await send_status_update(input, "started")

                job_input = input

                try:
                    queued_workflow = await queue_workflow_comfy_deploy(
                        job_input
                    )  # queue_workflow(workflow)
                    prompt_id = queued_workflow["prompt_id"]
                    print(f"comfy-modal - queued workflow with ID {prompt_id}")
                except Exception as e:
                    import traceback

                    print(traceback.format_exc())
                    raise e

                    # return {"error": f"Error queuing workflow: {str(e)}"}

                # Poll for completion
                print(f"comfy-modal - wait until image generation is complete")
                retries = 0
                status = ""
                try:
                    print("getting request")
                    while True:
                        status_result = await check_status(prompt_id=prompt_id)

                        # print("status_result", status_result)

                        if (
                            "status" in status_result
                            and status_result["status"] == "running"
                        ):
                            self.current_input = input
                            self.status_endpoint = input.status_endpoint

                        if "status" in status_result and (
                            status_result["status"] == "success"
                            or status_result["status"] == "failed"
                        ):
                            status = status_result["status"]
                            print(status)
                            break
                        else:
                            # Wait before trying again
                            await asyncio.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
                            retries += 1
                except Exception as e:
                    print(traceback.format_exc())
                    raise e
                    # return {"error": f"Error waiting for image generation: {str(e)}"}

                print(f"comfy-modal - Finished, turning off")

                result = {"status": status}

            except TimeoutError:
                print("Operation timed out")
                # return {"status": "failed"}
            except Exception as e:
                print(f"Unexpected error occurred: {str(e)}")
                await send_status_update(input, "failed")
                self.machine_logs.append({"logs": str(e), "timestamp": time.time()})
            finally:
                signal.alarm(0)

            print("uploading log_data")
            data = json.dumps(
                {
                    "run_id": input.prompt_id,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "log_data": self.machine_logs,
                }
            ).encode("utf-8")
            print("my logs", len(self.machine_logs))
            # Clear logs
            timeout = aiohttp.ClientTimeout(total=60)  # 60 seconds total timeout
            # Use HTTP/1.1 explicitly and increase the connection pool size
            connector = TCPConnector(
                limit=100, force_close=True, enable_cleanup_closed=True
            )

            async with aiohttp.ClientSession(
                timeout=timeout, connector=connector, trust_env=True
            ) as session:
                try:
                    async with session.post(
                        input.status_endpoint,
                        data=data,
                        headers={
                            "Content-Type": "application/json",
                            "bypass-tunnel-reminder": "true",
                            "Authorization": f"Bearer {config['auth_token']}",
                        },
                    ) as response:
                        print("response", response)
                        # Process your response here
                except asyncio.TimeoutError:
                    print("Request timed out")
                except Exception as e:
                    print(f"An error occurred: {e}")
            print("uploaded log_data")
            # print(data)
            self.machine_logs = []
        except asyncio.CancelledError as e:
            print("This got fucking cancelled", e)  # this doesn't happen
            try:
                ok = await interrupt_comfyui()
            except Exception as e:
                pass
            finally:
                await send_status_update(input, "cancelled")

        finally:
            # self.log_queues.pop(input.prompt_id)
            if len(self.log_queues) > 0:
                self.log_queues.pop()
            # if self.log_task:
            #     self.log_task.cancel()
            #     try:
            #         await self.log_task
            #     except asyncio.CancelledError:
            #         pass

        # commit by default so commit after the workflow is executed
        self.private_volume.commit()
        return result


HOST = "127.0.0.1"
PORT = "8188"


def spawn_comfyui_in_background(cpu_only: bool = False):
    import socket
    import subprocess

    commands = [
        "python",
        "main.py",
        "--listen",
        "0.0.0.0",  # need to allow local network connections for the tunnel to work
        "--dont-print-server",
        "--enable-cors-header",
        "--port",
        PORT,
    ]
    if cpu_only:
        commands.append("--cpu")
    process = subprocess.Popen(
        commands,
        cwd="/comfyui",
    )

    # Poll until webserver accepts connections before running inputs.
    while True:
        try:
            socket.create_connection((HOST, int(PORT)), timeout=1).close()
            print("ComfyUI webserver ready!")
            break
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"comfyui main.py exited unexpectedly with code {retcode}"
                )

    return process


class Body(BaseModel):
    comfy_deploy_url: str


CPU_TIMEOUT_SECONDS = 60 * 15  # 15 minutes
HRS24_SECONDS = 60 * 60 * 24


@app.cls(
    image=target_image,
    gpu=final_gpu_param,
    volumes=volumes,
    allow_concurrent_inputs=100,
    concurrency_limit=1,
    # keep_warm=0,
    timeout=1800,
    # _allow_background_volume_commits=True,
    secrets=modal_secrets,
)
class Workspace:
    @modal.web_server(
        8188, startup_timeout=120, label=machine_name_for_label + "-workspace"
    )
    def web(self):
        cmd = comfyui_cmd()
        #  + " --front-end-version Comfy-Org/ComfyUI_frontend@latest"
        subprocess.Popen(cmd, shell=True, cwd="/comfyui")

    @enter()
    async def setup(self):
        self.gpu_event_id = await sync_report_gpu_event(None)

    @exit()
    async def cleanup(self):
        await sync_report_gpu_event(self.gpu_event_id)


# Do not account for billing for cpu only
@app.cls(
    image=target_image,
    gpu=None,
    volumes=volumes,
    allow_concurrent_inputs=100,
    concurrency_limit=1,
    # keep_warm=0,
    timeout=1800,
    container_idle_timeout=1200,
    # _allow_background_volume_commits=True,
    secrets=modal_secrets,
)
class WorkspaceCPU:
    @modal.web_server(
        8188, startup_timeout=120, label=machine_name_for_label + "-ws-cpu"
    )
    def web(self):
        cmd = comfyui_cmd(True)
        subprocess.Popen(cmd, shell=True, cwd="/comfyui")


@app.function(
    image=image,
    gpu=None,
)
@web_endpoint(
    method="GET", wait_for_response=True, label=machine_name_for_label + "-app-id"
)
def get_app_id():
    return {"app_id": app.app_id}
