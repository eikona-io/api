from collections import deque
from typing import Optional, Dict, List, Any
from uuid import uuid4
from database import get_clickhouse_client
from models import Machine
from routes.utils import select
from pydantic import BaseModel, Field, field_validator
from fastapi import (
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Request,
)
from fastapi.responses import JSONResponse
import os
from enum import Enum
import json
import time
import aiohttp
import asyncio
import logging
import modal
from fastapi import APIRouter
import tempfile
from database import get_db
import logfire

machine_id_websocket_dict = {}
machine_id_websocket_dict_live_logs = {}
machine_id_status = {}
from fastapi import BackgroundTasks
import logging
from typing import List, Optional

from fastapi import Depends
from database import get_clickhouse_client
from clickhouse_connect.driver.asyncclient import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
import datetime as dt
from asyncio import Queue

# logger = logging.getLogger(__name__)

logfire.configure()
logger = logfire
logging.basicConfig(level=logging.INFO)

router = APIRouter(tags=["Machine"], prefix="/machine")

public_model_volume_name = os.environ.get(
    "PUBLIC_MODEL_VOLUME_NAME", os.environ.get("SHARED_MODEL_VOLUME_NAME", "local")
)
app_env = os.environ.get("APP_ENV", "production")


change_log_dict = {
    "0.1.2": "Initial release.",
    "0.1.3": "Fixed bugs related to websocket handling related to bytes and text.",
    "0.1.4": "Added back legacy comfy manager models install",
    "0.1.5": "Private models are mounted onto /comfyui/models instead of symlinking",
    "0.1.6": "Private models are back to symlinking",
    "0.1.7": "Extra model paths add ultralytics_bbox and ultralytics_segm, Added options to change concurrency limit and allow concurrent inputs",
    "0.1.8": "GPU usage stats",
    "0.1.9": "Inpainting model support",
    "0.1.10": "function call id, send on function run",
    "0.1.11": "Build dependencies on GPU",
    "0.1.12": "Webhook logs",
    "0.1.13": "fix: GPU build dependencies fully dependent on config",
    "0.2.0": "Workspace session, refactor machine support for custom inputs\nBreaking change: machine bundled images will not work due to the storage and inputs folder, please upload using workspace.",
    "0.2.1": "fix: building with GPU",
    "0.2.2": "fix: remove pydantic locked version\nfix: install fastapi version to 0.109.0",
    "0.2.3": "fix: log not showing up with consecutive run",
    "0.2.4": "fix: issues causing not-started runs in 0.2.3",
    "0.2.5": "fix: timezone issues",
    "0.2.6": "feat: change base docker image",
    "0.3.0": "feat: v3",
    # Add future version changes here
}
BUILDER_VERSION = list(change_log_dict.keys())[-1]


@router.get("/version")
def read_version():
    return {
        "version": BUILDER_VERSION,
        "changelog": change_log_dict.get(BUILDER_VERSION, "No change log available"),
    }


class GitCustomNodes(BaseModel):
    hash: str
    disabled: bool
    pip: Optional[List[str]] = None


class FileCustomNodes(BaseModel):
    filename: str
    disabled: bool


class Snapshot(BaseModel):
    comfyui: str
    git_custom_nodes: Dict[str, GitCustomNodes]
    file_custom_nodes: List[FileCustomNodes]


class Model(BaseModel):
    name: str
    type: str
    base: str
    save_path: str
    description: str
    reference: str
    filename: str
    url: str


class GPUType(str, Enum):
    T4 = "T4"
    A10G = "A10G"
    A100 = "A100"
    A100_80GB = "A100-80GB"
    H100 = "H100"
    L4 = "L4"


class Item(BaseModel):
    machine_id: str
    name: str
    auth_token: str
    snapshot: Optional[Snapshot] = None
    models: Optional[List[Model]] = None
    callback_url: str
    cd_callback_url: str
    gpu_event_callback_url: str
    model_volume_name: str
    run_timeout: Optional[int] = Field(default=60 * 5)
    idle_timeout: Optional[int] = Field(default=60)
    ws_timeout: Optional[int] = Field(default=2)
    legacy_mode: Optional[bool] = Field(default=False)
    install_custom_node_with_gpu: Optional[bool] = Field(default=False)
    gpu: GPUType = Field(default=GPUType.T4)
    concurrency_limit: Optional[int] = Field(default=2)
    allow_concurrent_inputs: Optional[int] = Field(default=50)
    deps: Optional[Any] = None
    docker_commands: Optional[List[List[str]]] = None
    machine_builder_version: Optional[str] = "2"
    allow_background_volume_commits: Optional[bool] = Field(default=False)
    skip_static_assets: Optional[bool] = Field(default=False)
    retrieve_static_assets: Optional[bool] = Field(default=False)
    base_docker_image: Optional[str] = Field(default="")
    python_version: Optional[str] = Field(default="")
    prestart_command: Optional[str] = Field(default="")
    extra_args: Optional[str] = Field(default="")
    modal_app_id: Optional[str] = None

    @field_validator("gpu")
    @classmethod
    def check_gpu(cls, value):
        if not isinstance(value, GPUType):
            try:
                return GPUType(value)
            except ValueError:
                raise ValueError(
                    f"Invalid GPU option. Choose from: {', '.join(GPUType.__members__)}"
                )
        return value


class KeepWarmBody(BaseModel):
    warm_pool_size: int = 1


# Keep warm for workspace app
@router.post("/modal/{app_name}/keep-warm")
def set_modal_keep_warm(app_name: str, body: KeepWarmBody):
    try:
        a = modal.Cls.lookup(app_name, "ComfyDeployRunner")
        print("Keep warm start", a)
        a().keep_warm(body.warm_pool_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error keep machine warm {str(e)}")

    print("Keep warm succuss", a)
    return {"status": "success"}


# @router.post("/modal/{app_name}/keep-warm-workspace")
# def set_modal_keep_warm_workspace(app_name: str, body: KeepWarmBody):
#     try:
#         a = modal.Cls.lookup(app_name, "Workspace")
#         a().web.keep_warm(body.warm_pool_size)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error keep machine warm {str(e)}")
#     return {"status": "success"}


class CancelFunctionBody(BaseModel):
    function_id: str


@router.post("/modal/cancel-function")
def modal_cancel_function(body: CancelFunctionBody):
    try:
        a = modal.functions.FunctionCall.from_id(body.function_id)
        # print(a)
        a.cancel()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancel function {str(e)}")
    return {"status": "success"}


@router.websocket("/ws/{machine_id}")
async def websocket_endpoint(
    # request: Request,
    websocket: WebSocket,
    machine_id: str,
    client: AsyncClient = Depends(get_clickhouse_client),
    db: AsyncSession = Depends(get_db),
):
    await websocket.accept()
    # machine_id_websocket_dict[machine_id] = websocket

    try:
        last_update_time = None

        machine = (
            await db.execute(
                select(Machine).where(Machine.id == machine_id)
                # .apply_org_check(request)
            )
        ).scalar_one_or_none()

        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")

        last_update_time = machine.updated_at

        while True:
            # Fetch logs from ClickHouse
            query = f"""
            SELECT timestamp, message
            FROM log_entries
            WHERE machine_id = '{machine_id}'
            {f"AND timestamp > toDateTime64('{last_update_time.isoformat()}', 6)" if last_update_time else ""}
            ORDER BY timestamp ASC
            LIMIT 100
            """
            result = await client.query(query)

            if result.result_rows:
                for row in result.result_rows:
                    timestamp, logs = row
                    last_update_time = timestamp

                    # Check if logs is a string or a list
                    if isinstance(logs, str):
                        try:
                            # Try to parse the string as JSON
                            log_entries = json.loads(logs)
                            if not isinstance(log_entries, list):
                                log_entries = [log_entries]
                        except json.JSONDecodeError:
                            # If parsing fails, treat it as a single log entry
                            log_entries = [logs]
                    elif isinstance(logs, list):
                        log_entries = logs
                    else:
                        # If it's neither string nor list, convert to string and wrap in a list
                        log_entries = [str(logs)]

                    # Process each log entry
                    for log_entry in log_entries:
                        log_data = {
                            "event": "LOGS",
                            "data": {
                                "machine_id": machine_id,
                                "logs": str(log_entry),
                                "timestamp": timestamp.timestamp(),
                            },
                        }
                        await websocket.send_text(json.dumps(log_data))
            else:
                # Send a keepalive message if no new logs
                await websocket.send_text(json.dumps({"event": "KEEPALIVE"}))

            # Wait for a short interval before the next poll
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    # except WebSocketDisconnect:
    #     if machine_id in machine_id_websocket_dict:
    #         machine_id_websocket_dict.pop(machine_id)
    # finally:
    #     if machine_id in machine_id_websocket_dict:
    #         machine_id_websocket_dict.pop(machine_id)


@router.websocket("/ws/{machine_id}/live-logs")
async def websocket_endpoint_live_logs(websocket: WebSocket, machine_id: str):
    await websocket.accept()
    machine_id_websocket_dict_live_logs[machine_id] = websocket

    task = asyncio.create_task(listen_logs(machine_id=machine_id))

    try:
        while True:
            data = await websocket.receive_text()
            # global last_activity_time
            # last_activity_time = time.time()
            # logger.info(f"Extended inactivity time to {global_timeout}")
            # You can handle received messages here if needed
    except WebSocketDisconnect:
        if machine_id in machine_id_websocket_dict_live_logs:
            machine_id_websocket_dict_live_logs.pop(machine_id)

        task.cancel()


@router.post("/create")
async def create_machine(
    item: Item,
    background_tasks: BackgroundTasks,
):
    logger.info("Application starting up hahahahahaha")
    
    if item.machine_id in machine_id_status and machine_id_status[item.machine_id]:
        return JSONResponse(
            status_code=400, content={"error": "Build already in progress."}
        )

    # Run the building logic in a separate thread
    # future = executor.submit(build_logic, item)
    background_tasks.add_task(build_logic, item)

    return JSONResponse(
        status_code=200,
        content={
            "message": "Build Queued",
            "build_machine_instance_id": "not-applicable",
        },
    )


class StopAppItem(BaseModel):
    machine_id: str


def find_app_id(app_list, app_name):
    for app in app_list:
        if app["Name"] == app_name:
            return app["App ID"]
    return None


@router.post("/stop-app")
async def stop_app(item: StopAppItem):
    # cmd = f"modal app list | grep {item.machine_id} | awk -F '│' '{{print $2}}'"
    cmd = f"modal app list --json"

    env = os.environ.copy()
    env["COLUMNS"] = "10000"  # Set the width to a large value
    find_id_process = await asyncio.subprocess.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
    )
    await find_id_process.wait()

    stdout, stderr = await find_id_process.communicate()
    if stdout:
        app_id = stdout.decode().strip()
        app_list = json.loads(app_id)
        app_id = find_app_id(app_list, item.machine_id)
        logger.info(f"cp_process stdout: {app_id}")
    if stderr:
        logger.info(f"cp_process stderr: {stderr.decode()}")

    cp_process = await asyncio.subprocess.create_subprocess_exec(
        "modal",
        "app",
        "stop",
        app_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await cp_process.wait()
    logger.info(f"Stopping app {item.machine_id}")
    stdout, stderr = await cp_process.communicate()
    if stdout:
        logger.info(f"cp_process stdout: {stdout.decode()}")
    if stderr:
        logger.info(f"cp_process stderr: {stderr.decode()}")

    if cp_process.returncode == 0:
        return JSONResponse(status_code=200, content={"status": "success"})
    else:
        return JSONResponse(
            status_code=500, content={"status": "error", "error": stderr.decode()}
        )


# Initialize the logs cache
machine_logs_cache = {}


async def listen_logs(machine_id):
    process = await asyncio.subprocess.create_subprocess_shell(
        "modal app logs " + machine_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "COLUMNS": "10000"},
    )

    async def read_stream(stream, isStderr):
        buffer = []
        last_send_time = time.time()
        buffer_interval = 5  # seconds

        while True:
            line = await stream.readline()
            if line:
                l = line.decode("utf-8").strip()
                if l:
                    buffer.append(l)
                    current_time = time.time()
                    if current_time - last_send_time >= buffer_interval:
                        if machine_id in machine_id_websocket_dict_live_logs:
                            await machine_id_websocket_dict_live_logs[
                                machine_id
                            ].send_text(
                                json.dumps(
                                    {
                                        "event": "LOGS",
                                        "data": {
                                            "machine_id": machine_id,
                                            "logs": "\n".join(buffer),
                                            "timestamp": current_time,
                                        },
                                    }
                                )
                            )
                        buffer = []
                        last_send_time = current_time
            else:
                break

        # Send any remaining logs in the buffer
        if buffer:
            if machine_id in machine_id_websocket_dict_live_logs:
                await machine_id_websocket_dict_live_logs[machine_id].send_text(
                    json.dumps(
                        {
                            "event": "LOGS",
                            "data": {
                                "machine_id": machine_id,
                                "logs": "\n".join(buffer),
                                "timestamp": time.time(),
                            },
                        }
                    )
                )

    stdout_task = asyncio.create_task(read_stream(process.stdout, False))
    stderr_task = asyncio.create_task(read_stream(process.stderr, True))

    await asyncio.wait([stdout_task, stderr_task])

    # Wait for the subprocess to finish
    await process.wait()


async def send_json_to_ws(machine_id, event, data):
    if machine_id in machine_id_websocket_dict:
        await machine_id_websocket_dict[machine_id].send_text(
            json.dumps(
                {
                    "event": event,
                    "data": data,
                }
            )
        )


async def insert_to_clickhouse(table: str, data: list):
    # logger.info(f"Inserting to clickhouse: {table}, {data}")
    client = await get_clickhouse_client()
    await client.insert(table=table, data=data)


async def build_logic(item: Item):
    # Deploy to modal
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path = temp_dir

        # os.makedirs("./app/builds", exist_ok=True)

        folder_path = f"{folder_path}/{item.machine_id}"
        machine_id_status[item.machine_id] = True

        # Ensure the os path is same as the current directory
        # os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # print(
        #     f"builder - Current working directory: {os.getcwd()}"
        # )

        # Copy the app template
        # os.system(f"cp -r template {folder_path}")
        print(
            f"builder - VSCODE_DEV_CONTAINER: {os.environ.get('VSCODE_DEV_CONTAINER', 'false')}"
        )
        version_to_file = item.machine_builder_version  # .replace(".", "_")
        to_modal_deps_version = {"2": None, "3": "2024.04"}
        cp_process = await asyncio.subprocess.create_subprocess_exec(
            "cp",
            "-r",
            f"./src/api/modal/v{version_to_file}",
            folder_path,
        )

        print(os.listdir("."))

        print("copying file, ", folder_path)
        await cp_process.wait()
        print("copied file")

        pip_modules = set()
        if item.snapshot != None:
            for git_custom_node in item.snapshot.git_custom_nodes.values():
                if git_custom_node.pip:
                    pip_modules.update(git_custom_node.pip)

        # Write the config file
        config = {
            "name": item.name,
            "deploy_test": "False",
            "gpu": item.gpu,
            "public_model_volume": public_model_volume_name,
            "private_model_volume": item.model_volume_name,
            "pip": list(pip_modules),
            "run_timeout": item.run_timeout,
            "idle_timeout": 2 if item.idle_timeout < 2 else item.idle_timeout,
            "ws_timeout": item.ws_timeout,
            "machine_id": item.machine_id,
            "auth_token": item.auth_token,
            "gpu_event_callback_url": item.gpu_event_callback_url,
            "cd_callback_url": item.cd_callback_url,
            "legacy_mode": "True" if item.legacy_mode else "False",
            "allow_concurrent_inputs": item.allow_concurrent_inputs,
            "concurrency_limit": item.concurrency_limit,
            "install_custom_node_with_gpu": "True"
            if item.install_custom_node_with_gpu
            else "False",
            "allow_background_volume_commits": "True"
            if item.allow_background_volume_commits
            else "False",
            "skip_static_assets": "True" if item.skip_static_assets else "False",
            "base_docker_image": item.base_docker_image,
            "python_version": item.python_version,
            "prestart_command": item.prestart_command,
            "extra_args": item.extra_args,
        }

        # print("config: ", config)

        config_file_path = os.path.abspath(f"{folder_path}/config.py")
        print(f"Debug: Opening config file at: {config_file_path}")

        if os.environ.get("TAILSCALE_AUTHKEY", None) is not None:
            config["TAILSCALE_AUTHKEY"] = os.environ.get("TAILSCALE_AUTHKEY", None)

        if item.deps != None:
            config["deps"] = item.deps

        if item.docker_commands != None:
            config["docker_commands"] = item.docker_commands

        with open(f"{folder_path}/config.py", "w") as f:
            f.write("config = " + json.dumps(config))

        if item.snapshot != None:
            with open(f"{folder_path}/data/snapshot.json", "w") as f:
                f.write(item.snapshot.json())

        with open(f"{folder_path}/data/models.json", "w") as f:
            if item.models != None:
                models_json_list = [model.dict() for model in item.models]
                models_json_string = json.dumps(models_json_list)
                f.write(models_json_string)
            else:
                f.write("[]")

        # os.chdir(folder_path)
        # process = subprocess.Popen(f"modal deploy {folder_path}/app.py", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        final_env = {
            **os.environ,
            "COLUMNS": "10000",
        }
        if to_modal_deps_version[version_to_file]:
            final_env["MODAL_IMAGE_BUILDER_VERSION"] = to_modal_deps_version[
                version_to_file
            ]

        process = await asyncio.subprocess.create_subprocess_shell(
            "modal deploy my_app.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=folder_path,
            env=final_env,
        )

        url = None

        if item.machine_id not in machine_logs_cache:
            machine_logs_cache[item.machine_id] = []

        machine_logs = machine_logs_cache[item.machine_id]

        build_info_queue = asyncio.Queue()

        # Initialize a buffer for logs
        log_buffer = deque()
        FLUSH_INTERVAL = 1  # Flush every 5 seconds
        buffer_lock = asyncio.Lock()
        flush_task = None

        async def flush_log_buffer():
            nonlocal flush_task, log_buffer
            async with buffer_lock:
                if log_buffer:
                    await insert_to_clickhouse("log_entries", list(log_buffer))
                    log_buffer.clear()

        async def periodic_flush():
            while True:
                await asyncio.sleep(FLUSH_INTERVAL)
                await flush_log_buffer()

        async def add_to_log_buffer(log_entry):
            nonlocal flush_task, log_buffer
            async with buffer_lock:
                log_buffer.append(log_entry)

            if flush_task is None or flush_task.done():
                flush_task = asyncio.create_task(periodic_flush())

        async def read_stream(stream, isStderr, build_info_queue: asyncio.Queue):
            while True:
                line = await stream.readline()
                if line:
                    l = line.decode("utf-8").strip()

                    if l == "":
                        continue

                    updated_at = dt.datetime.now(dt.UTC)
                    log_entry = (
                        uuid4(),
                        None,
                        None,
                        item.machine_id,
                        updated_at,
                        "builder",
                        l,
                    )
                    await add_to_log_buffer(log_entry)

                    if not isStderr:
                        logger.info(l)
                        machine_logs.append({"logs": l, "timestamp": time.time()})

                        if l.startswith("APPID="):
                            app_id = l.split("=")[1].strip()
                            await build_info_queue.put(
                                {"type": "APPID", "value": app_id}
                            )

                        if (
                            "Created comfyui_api =>" in l
                            or "Created ComfyDeployRunner.api =>" in l
                            or "Created web function comfyui_api =>" in l
                        ) or (
                            (l.startswith("https://") or l.startswith("│"))
                            and l.endswith(".modal.run")
                        ):
                            if (
                                "Created comfyui_api =>" in l
                                or "Created ComfyDeployRunner.api =>" in l
                                or "Created web function comfyui_api =>" in l
                            ):
                                url = l.split("=>")[1].strip()
                            # making sure it is a url
                            elif "comfyui-api" in l:
                                # Some case it only prints the url on a blank line
                                if l.startswith("│"):
                                    url = l.split("│")[1].strip()
                                else:
                                    url = l

                            if url:
                                machine_logs.append(
                                    {
                                        "logs": f"App image built, url: {url}",
                                        "timestamp": time.time(),
                                    }
                                )

                                # await url_queue.put(url)
                                await build_info_queue.put(
                                    {"type": "URL", "value": url}
                                )

                                await send_json_to_ws(
                                    item.machine_id,
                                    "LOGS",
                                    {
                                        "machine_id": item.machine_id,
                                        "logs": f"App image built, url: {url}",
                                        "timestamp": time.time(),
                                    },
                                )
                                # await machine_id_websocket_dict[
                                #     item.machine_id
                                # ].send_text(
                                #     json.dumps(
                                #         {
                                #             "event": "FINISHED",
                                #             "data": {
                                #                 "status": "succuss",
                                #             },
                                #         }
                                #     )
                                # )

                    else:
                        # is error
                        logger.error(l)
                        machine_logs.append({"logs": l, "timestamp": time.time()})
                else:
                    break

            # Ensure any remaining logs in the buffer are flushed
            await flush_log_buffer()
            if flush_task:
                flush_task.cancel()

        stdout_task = asyncio.create_task(
            read_stream(process.stdout, False, build_info_queue)
        )
        stderr_task = asyncio.create_task(
            read_stream(process.stderr, True, build_info_queue)
        )

        await asyncio.wait([stdout_task, stderr_task])

        # Wait for the subprocess to finish
        await process.wait()

        url = None
        app_id = None
        while not build_info_queue.empty():
            info = await build_info_queue.get()
            if info["type"] == "APPID":
                app_id = info["value"]
            elif info["type"] == "URL":
                url = info["value"]

        # Replace 'comfyui-api' in the URL with 'app-id' and parse the returned JSON to get app_id
        if url and item.modal_app_id is None:
            appid_url = url.replace("comfyui-api", "app-id")
            async with aiohttp.ClientSession() as session:
                async with session.get(appid_url) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        app_id = response_json.get("app_id", None)
                        if app_id:
                            logger.info(f"App ID: {app_id}")
                        else:
                            logger.error("App ID not found in the response.")
                    else:
                        logger.error(
                            f"Failed to fetch App ID. HTTP Status: {response.status}"
                        )
        
        if item.modal_app_id:
            app_id = item.modal_app_id

    async def clear_machine_logs_and_remove_folder(machine_id, folder_path):
        # Close the ws connection and also pop the item
        if (
            item.machine_id in machine_id_websocket_dict
            and machine_id_websocket_dict[item.machine_id] is not None
        ):
            await machine_id_websocket_dict[item.machine_id].close()

        if item.machine_id in machine_id_websocket_dict:
            machine_id_websocket_dict.pop(item.machine_id)

        if item.machine_id in machine_id_status:
            machine_id_status[item.machine_id] = False

        if machine_id in machine_logs_cache:
            del machine_logs_cache[machine_id]

        import shutil

        # Remove the copied folder
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Removed folder: {folder_path}")

    # Check for errors
    if process.returncode != 0:
        logger.info("An error occurred.")
        # Send a post request with the json body machine_id to the callback url
        machine_logs.append(
            {"logs": "Unable to build the app image.", "timestamp": time.time()}
        )
        async with aiohttp.ClientSession() as session:
            await session.post(
                item.callback_url,
                headers={
                    "Content-Type": "application/json",
                    "bypass-tunnel-reminder": "true",
                },
                data=json.dumps(
                    {
                        "machine_id": item.machine_id,
                        "build_log": json.dumps(machine_logs),
                    }
                ).encode("utf-8"),
            )
        await clear_machine_logs_and_remove_folder(item.machine_id, folder_path)
        return

    if url is None:
        machine_logs.append(
            {
                "logs": "App image built, but url is None, unable to parse the url.",
                "timestamp": time.time(),
            }
        )
        async with aiohttp.ClientSession() as session:
            await session.post(
                item.callback_url,
                headers={
                    "Content-Type": "application/json",
                    "bypass-tunnel-reminder": "true",
                },
                data=json.dumps(
                    {
                        "machine_id": item.machine_id,
                        "build_log": json.dumps(machine_logs),
                    }
                ).encode("utf-8"),
            )
        await clear_machine_logs_and_remove_folder(item.machine_id, folder_path)
        return

    print("callback_url", item.callback_url)
    async with aiohttp.ClientSession() as session:
        await session.post(
            item.callback_url,
            headers={
                "Content-Type": "application/json",
                "bypass-tunnel-reminder": "true",
            },
            data=json.dumps(
                {
                    "machine_id": item.machine_id,
                    "endpoint": url,
                    "app_id": app_id,
                    "build_log": json.dumps(machine_logs),
                    # "static_fe_assets": static_fe_assets,
                }
            ).encode("utf-8"),
        )

    await send_json_to_ws(
        item.machine_id,
        "FINISHED",
        {
            "status": "succuss",
        },
    )

    await clear_machine_logs_and_remove_folder(item.machine_id, folder_path)
