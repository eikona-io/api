import asyncio
from datetime import datetime
import os
from pprint import pprint
from api.modal.builder import insert_to_clickhouse
from api.routes.machines import (
    UpdateServerlessMachineModel,
    redeploy_machine,
    update_serverless_machine,
)
from api.routes.types import GPUEventModel, MachineGPU, MachineType
from api.utils.docker import (
    CustomNode,
    DepsBody,
    DockerStep,
    DockerSteps,
    comfyui_cmd,
    generate_all_docker_commands,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import modal
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import select
from pydantic import BaseModel, Field

# from sqlalchemy import select
from api.models import (
    GPUEvent,
    Machine,
    MachineVersion,
    get_machine_columns,
)
from api.database import get_clickhouse_client, get_db, get_db_context
from typing import Any, Dict, List, Optional, cast, Union
from uuid import UUID, uuid4
import logging
from typing import Optional
from sqlalchemy import update, func
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Session"])
beta_router = APIRouter(tags=["Beta"])

status_endpoint = os.environ.get("CURRENT_API_URL") + "/api/update-run"


async def get_comfy_runner(
    machine_id: str, session_id: str | UUID, timeout: int, gpu: str
):
    logger.info(machine_id)
    ComfyDeployRunner = await modal.Cls.lookup.aio(str(machine_id), "ComfyDeployRunner")
    runner = ComfyDeployRunner.with_options(
        concurrency_limit=1,
        allow_concurrent_inputs=1000,
        # 2 seconds minimum idle timeout
        container_idle_timeout=2,
        timeout=timeout * 60,
        gpu=gpu if gpu != "CPU" else None,
    )(session_id=str(session_id), gpu=gpu)

    return runner


class Session(BaseModel):
    session_id: str
    gpu_event_id: str
    url: Optional[str]
    gpu: str
    created_at: datetime
    timeout: Optional[int]
    machine_id: Optional[str]


# Return the session tunnel url
@router.get(
    "/session/{session_id}",
    openapi_extra={
        "x-speakeasy-name-override": "get",
    },
)
async def get_session(
    request: Request, session_id: str, db: AsyncSession = Depends(get_db)
) -> Session:
    gpuEvent = cast(
        Optional[GPUEvent],
        (
            await db.execute(
                (
                    select(GPUEvent)
                    .where(GPUEvent.session_id == session_id)
                    .where(GPUEvent.end_time.is_(None))
                    .apply_org_check(request)
                )
            )
        ).scalar_one_or_none(),
    )

    if gpuEvent is None:
        raise HTTPException(status_code=404, detail="GPUEvent not found")

    # runner = get_comfy_runner(gpuEvent.machine_id, gpuEvent.session_id)

    # async with modal.Queue.ephemeral() as q:
    #     await runner.create_tunnel.spawn.aio(q, status_endpoint)
    #     url = await q.get.aio()

    return {
        "session_id": session_id,
        "url": gpuEvent.tunnel_url,
        "gpu_event_id": str(gpuEvent.id),
        "gpu": gpuEvent.gpu,
        "created_at": gpuEvent.created_at,
        "timeout": gpuEvent.session_timeout,
        "machine_id": str(gpuEvent.machine_id) if gpuEvent.machine_id else None,
    }
    # with logfire.span("spawn-run"):
    #     result = ComfyDeployRunner().run.spawn(params)
    #     new_run.modal_function_call_id = result.object_id


class GetSessionsBody(BaseModel):
    machine_id: str


# Return the sessions for a machine
@router.get(
    "/sessions",
    openapi_extra={
        "x-speakeasy-name-override": "list",
    },
)
async def get_machine_sessions(
    request: Request,
    machine_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> List[GPUEventModel]:
    query = (
        select(GPUEvent)
        .where(GPUEvent.end_time.is_(None))
        .where(GPUEvent.session_id.isnot(None))
        .order_by(GPUEvent.start_time.desc())
        .apply_org_check(request)
    )

    if machine_id:
        query = query.where(GPUEvent.machine_id == machine_id)

    result = await db.execute(query)
    return result.scalars().all()


async def increase_timeout_task(
    machine_id: str, session_id: UUID, timeout: int, gpu: str
):
    runner = await get_comfy_runner(machine_id, session_id, 60 * 24, gpu)
    await runner.increase_timeout.spawn.aio(timeout)


async def create_session_background_task(
    machine_id: str,
    session_id: UUID,
    request: Request,
    timeout: int,
    gpu: str,
    status_queue: Optional[asyncio.Queue] = None,
):
    runner = await get_comfy_runner(machine_id, session_id, 60 * 24, gpu)

    try:
        runner.increase_timeout
        has_increase_timeout = True
    except (AttributeError, modal.exception.Error):
        has_increase_timeout = False

    if not has_increase_timeout:
        runner = await get_comfy_runner(machine_id, session_id, timeout, gpu)

    print("async_creation", status_queue)
    async with modal.Queue.ephemeral() as q:
        if has_increase_timeout:
            result = await runner.create_tunnel.spawn.aio(q, status_endpoint, timeout)
        else:
            result = await runner.create_tunnel.spawn.aio(q, status_endpoint)

        modal_function_id = result.object_id
        print("modal_function_id", modal_function_id)

        # gpuEvent = None
        # while gpuEvent is None:
        # async with get_db_context() as db:
        #     gpuEvent = cast(
        #         Optional[GPUEvent],
        #         (
        #             await db.execute(
        #                 (
        #                     select(GPUEvent)
        #                     .where(GPUEvent.session_id == str(session_id))
        #                     .where(GPUEvent.end_time.is_(None))
        #                     .apply_org_check(request)
        #                 )
        #             )
        #         ).scalar_one_or_none(),
        #     )

        # if gpuEvent is None:
        #     await asyncio.sleep(1)

        print("async_creation", status_queue)
        if status_queue is not None:
            await status_queue.put(modal_function_id)
        print("async_creation", modal_function_id)

        async with get_db_context() as db:
            result = await db.execute(
                update(GPUEvent)
                .where(GPUEvent.session_id == str(session_id))
                .values(modal_function_id=modal_function_id, session_timeout=timeout)
                .returning(GPUEvent)
            )
            await db.commit()
            gpuEvent = result.scalar_one()
            await db.refresh(gpuEvent)

        while True:
            msg = await q.get.aio()
            if msg.startswith("url:"):
                url = msg[4:]
                break
            else:
                logger.info(msg)

        async with get_db_context() as db:
            result = await db.execute(
                update(GPUEvent)
                .where(GPUEvent.session_id == str(session_id))
                .values(tunnel_url=url)
                .returning(GPUEvent)
            )
            await db.commit()
            gpuEvent = result.scalar_one()
            print("gpuEvent", gpuEvent)
            await db.refresh(gpuEvent)
            return gpuEvent


class CreateSessionBody(BaseModel):
    machine_id: str
    gpu: Optional[MachineGPU] = Field(None, description="The GPU to use")
    timeout: Optional[int] = Field(None, description="The timeout in minutes")
    wait_for_server: bool = Field(
        False, description="Whether to create the session asynchronously"
    )


class CreateDynamicSessionBody(BaseModel):
    gpu: MachineGPU = Field("A10G", description="The GPU to use")
    machine_id: Optional[str] = Field(None, description="The machine id to use")
    timeout: Optional[int] = Field(None, description="The timeout in minutes")
    dependencies: Optional[Union[List[str], DepsBody]] = Field(
        [],
        description="The dependencies to use, either as a DepsBody or a list of shorthand strings",
        examples=[
            [
                "Stability-AI/ComfyUI-SAI_API@1793086",
                "cubiq/ComfyUI_IPAdapter_plus@b188a6c",
            ]
        ],
    )
    wait_for_server: bool = Field(
        False, description="Whether to create the session asynchronously"
    )


async def ensure_session_creation_complete(task: asyncio.Task):
    try:
        await task
    except Exception as e:
        # Handle any exceptions that occurred during session creation
        logger.error(f"Session creation failed: {str(e)}")
        # You might want to update the session status in the database here


class IncreaseTimeoutBody(BaseModel):
    machine_id: str
    session_id: UUID
    timeout: int
    gpu: str


@router.post("/session/increase-timeout")
async def increase_timeout(
    request: Request, body: IncreaseTimeoutBody, db: AsyncSession = Depends(get_db)
):
    gpu_event = (
        await db.execute(
            select(GPUEvent)
            .where(GPUEvent.session_id == str(body.session_id))
            .where(GPUEvent.end_time.is_(None))
            .apply_org_check(request)
        )
    ).scalar_one_or_none()

    if gpu_event is None:
        raise HTTPException(status_code=404, detail="GPU event not found")

    await increase_timeout_task(
        body.machine_id, body.session_id, body.timeout, body.gpu
    )

    # in the database we save the timeout in minutes
    gpu_event.session_timeout = gpu_event.session_timeout + body.timeout
    await db.commit()

    return JSONResponse(
        status_code=200, content={"message": "Timeout increased successfully"}
    )


class CreateSessionResponse(BaseModel):
    session_id: UUID
    url: Optional[str] = None


# Create a new session for a machine, return the session id and url
@router.post(
    "/session",
    openapi_extra={
        "x-speakeasy-name-override": "create",
    },
)
async def create_session(
    request: Request,
    body: CreateSessionBody,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    org_id = request.state.current_user.get("org_id")
    user_id = request.state.current_user.get("user_id")
    # check if the user has reached the spend limit
    # exceed_spend_limit = await is_exceed_spend_limit(request, db)
    # if exceed_spend_limit:
    #     raise HTTPException(status_code=400, detail="Spend limit reached")

    machine_id = body.machine_id
    machine = cast(
        Optional[Machine],
        (
            await db.execute(
                (
                    select(Machine)
                    .where(Machine.id == machine_id)
                    .apply_org_check(request)
                )
            )
        ).scalar_one_or_none(),
    )

    if machine is None:
        raise HTTPException(status_code=404, detail="Machine not found")

    if machine.type != "comfy-deploy-serverless":
        raise HTTPException(
            status_code=400,
            detail="Machine is not a Comfy Deploy Serverless machine",
        )

    if int(machine.machine_builder_version) < 4:
        raise HTTPException(
            status_code=400,
            detail="Machine builder version is not larger than 4",
        )

    session_id = uuid4()

    gpu_event = GPUEvent(
        id=uuid4(),
        user_id=user_id,
        org_id=org_id,
        machine_id=machine_id,
        gpu=body.gpu.value if body.gpu is not None else machine.gpu,
        gpu_provider="modal",
        session_id=str(session_id),
        session_timeout=body.timeout or 15,
    )

    db.add(gpu_event)
    await db.commit()

    # Add the background task
    try:
        if not body.wait_for_server:
            print("async_creation")
            q = asyncio.Queue()

            task = asyncio.create_task(
                create_session_background_task(
                    machine_id,
                    session_id,
                    request,
                    body.timeout or 15,
                    body.gpu.value if body.gpu is not None else machine.gpu,
                    q,
                )
            )

            background_tasks.add_task(
                ensure_session_creation_complete,
                task,
            )

            try:
                modal_function_id = await asyncio.wait_for(q.get(), timeout=10.0)
                print("async_creation", "modal_function_id", modal_function_id)
            except asyncio.TimeoutError:
                print("Timed out waiting for modal_function_id")
                modal_function_id = None
        else:
            gpuEvent = await create_session_background_task(
                machine_id,
                session_id,
                request,
                body.timeout or 15,
                str(body.gpu.value if body.gpu is not None else machine.gpu),
            )

            return {
                "session_id": session_id,
                "url": gpuEvent.tunnel_url,
            }
    except Exception as e:
        logger.error(f"Session creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Session creation failed")

    return {
        "session_id": session_id,
    }


@beta_router.post(
    "/deps",
    openapi_extra={
        "x-speakeasy-name-override": "generateDockerSteps",
    },
)
async def convert_to_docker_steps(
    body: DepsBody,
):
    converted = generate_all_docker_commands(body)

    return JSONResponse(status_code=200, content=converted)


def extract_hash(dependency_string):
    parts = dependency_string.split("@")
    if len(parts) > 1:
        return parts[-1]
    return ""


def extract_url(dependency_string):
    parts = dependency_string.split("@")
    if len(parts) > 1:
        return "https://github.com/" + parts[0]
    return ""


async def create_dynamic_sesssion_background_task(
    request: Request,
    session_id: UUID,
    body: CreateDynamicSessionBody,
    status_queue: Optional[asyncio.Queue] = None,
):
    app = modal.App(str(session_id))
    # app = modal.App.lookup("dynamic-comfyui", create_if_missing=True)

    org_id = request.state.current_user.get("org_id")
    user_id = request.state.current_user.get("user_id")

    # Get machine and its version info
    modal_image_id = None
    if body.machine_id is not None:
        async with get_db_context() as db:
            machine = await db.execute(
                select(Machine)
                .where(Machine.id == body.machine_id)
                .apply_org_check(request)
            )
            machine = machine.scalars().first()

            if not machine:
                raise HTTPException(status_code=404, detail="Machine not found")

            # Get machine version if it exists
            if machine.machine_version_id:
                machine_version = await db.execute(
                    select(MachineVersion).where(
                        MachineVersion.id == machine.machine_version_id
                    )
                )
                machine_version = machine_version.scalars().first()

                if machine_version and machine_version.modal_image_id:
                    # Use existing modal image ID from version
                    modal_image_id = machine_version.modal_image_id

            if not modal_image_id:
                logger.info("No dependencies, using machine current settings")
                body.dependencies = DepsBody(
                    comfyui_version=machine.comfyui_version,
                    docker_command_steps=machine.docker_command_steps,
                )

    logger.info("Dependencies configuration " + str(body.dependencies))

    gpu_event_id = str(uuid4())

    logger.info("creating dynamic session")

    if (
        not modal_image_id and body.dependencies is not None
    ) or body.machine_id is None:
        print(body.machine_id)

        if body.dependencies is None:
            body.dependencies = []

        if isinstance(body.dependencies, list):
            # Handle shorthand dependencies
            deps_body = DepsBody(
                docker_command_steps=DockerSteps(
                    steps=[
                        DockerStep(
                            type="custom-node",
                            data=CustomNode(
                                install_type="git-clone",
                                url=extract_url(dep),
                                hash=extract_hash(dep),
                                name=dep.split("/")[-1],
                            ),
                        )
                        for dep in body.dependencies
                    ]
                )
            )
            converted = generate_all_docker_commands(
                deps_body, include_comfyuimanager=True
            )
        else:
            converted = generate_all_docker_commands(
                body.dependencies, include_comfyuimanager=True
            )

        # pprint(converted)

        dockerfile_image = modal.Image.debian_slim(python_version="3.11")

        docker_commands = converted.docker_commands
        if docker_commands is not None:
            for commands in docker_commands:
                dockerfile_image = dockerfile_image.dockerfile_commands(
                    commands,
                )

        dockerfile_image = dockerfile_image.run_commands(
            [
                "rm -rf /private_models /comfyui/models /public_models",
                "ln -s /private_models /comfyui/models",
            ]
        )

        current_directory = os.path.dirname(os.path.realpath(__file__))

        dockerfile_image = dockerfile_image.add_local_file(
            current_directory + "/extra_model_paths.yaml",
            "/comfyui/extra_model_paths.yaml",
        )
    else:
        logger.info(f"Using existing modal image {modal_image_id}")
        dockerfile_image = modal.Image.from_id(modal_image_id)

    if not dockerfile_image and body.dependencies is None:
        raise HTTPException(
            status_code=400, detail="No dependencies or modal image id provided"
        )

    gpu_event_id = str(uuid4())

    logger.info("creating dynamic session")

    try:
        async with app.run.aio():
            print("creating sandbox")
            print(dockerfile_image)
            sb = await modal.Sandbox.create.aio(
                # "bash",
                # "-c",
                # comfyui_cmd(cpu=True if body.gpu == "CPU" else False),
                image=dockerfile_image,
                timeout=(body.timeout or 15) * 60,
                gpu=body.gpu.value
                if body.gpu is not None and body.gpu.value != "CPU"
                else None,
                app=app,
                workdir="/comfyui",
                encrypted_ports=[8188],
                volumes={
                    "/public_models": modal.Volume.from_name(
                        os.environ.get("SHARED_MODEL_VOLUME_NAME")
                    ),
                    "/private_models": modal.Volume.from_name(
                        "models_" + org_id if org_id is not None else user_id
                    ),
                },
            )

            logger.info("creating gpu event")

            async with get_db_context() as db:
                # Insert GPU event
                new_gpu_event = GPUEvent(
                    id=gpu_event_id,
                    user_id=user_id,
                    org_id=org_id,
                    session_id=str(session_id),
                    machine_id=body.machine_id,
                    gpu=body.gpu.value if body.gpu is not None else "CPU",
                    session_timeout=body.timeout or 15,
                    gpu_provider="modal",
                    start_time=datetime.now(),
                    modal_function_id=sb.object_id,
                )
                db.add(new_gpu_event)
                await db.commit()
                await db.refresh(new_gpu_event)

            # logger.info(sb.tunnels())
            tunnels = await sb.tunnels.aio()
            tunnel = tunnels[8188]  # Access the tunnel after awaiting

            async with get_db_context() as db:
                await db.execute(
                    update(GPUEvent)
                    .where(GPUEvent.id == gpu_event_id)
                    .values(tunnel_url=tunnel.url)
                )
                await db.commit()

            await status_queue.put(tunnel.url)

            p = await sb.exec.aio(
                "bash",
                "-c",
                comfyui_cmd(
                    cpu=True if body.gpu == "CPU" else False,
                    install_latest_comfydeploy=True,
                ),
            )
            logger.info(tunnel.url)

            # async with await get_clickhouse_client() as client:
            try:

                async def log_stream(stream, stream_type: str):
                    async for line in stream:
                        try:
                            # Try to decode as UTF-8, replace invalid characters
                            if isinstance(line, bytes):
                                # Use 'ignore' instead of 'replace' to skip problematic characters
                                line = line.decode("utf-8", errors="ignore")

                                # Skip progress bar lines that contain these special characters
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
                                    ]
                                ):
                                    continue

                            print(line, end="")
                            data = [
                                (
                                    uuid4(),
                                    session_id,
                                    None,
                                    body.machine_id,
                                    datetime.now(),
                                    stream_type,
                                    line,
                                )
                            ]
                            asyncio.create_task(
                                insert_to_clickhouse("log_entries", data)
                            )
                        except Exception as e:
                            logger.error(f"Error processing log line: {str(e)}")

                # Create tasks for both stdout and stderr
                stdout_task = asyncio.create_task(log_stream(p.stdout, "info"))
                stderr_task = asyncio.create_task(log_stream(p.stderr, "info"))

                # Wait for both streams to complete
                await asyncio.gather(stdout_task, stderr_task)
            except Exception as e:
                logger.error(f"Error creating tasks: {str(e)}")

            await sb.wait.aio()
    except Exception as e:
        pass
    finally:
        async with get_db_context() as db:
            gpu_event = await db.execute(
                select(GPUEvent).where(GPUEvent.id == gpu_event_id)
            )
            gpu_event = gpu_event.scalar_one_or_none()
            gpu_event.end_time = datetime.now()
            await db.commit()
            await db.refresh(gpu_event)


@beta_router.post(
    "/session/dynamic",
    openapi_extra={
        "x-speakeasy-name-override": "createDynamic",
    },
)
async def create_dynamic_session(
    request: Request,
    body: CreateDynamicSessionBody,
    background_tasks: BackgroundTasks,
    # db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    session_id = uuid4()
    q = asyncio.Queue()

    task = asyncio.create_task(
        create_dynamic_sesssion_background_task(request, session_id, body, q)
    )

    background_tasks.add_task(
        ensure_session_creation_complete,
        task,
    )

    try:
        tunnel_url = await asyncio.wait_for(q.get(), timeout=300.0)
    except asyncio.TimeoutError:
        print("Timed out waiting for tunnel_url")
        tunnel_url = None

    return {
        "session_id": session_id,
        "url": tunnel_url,
    }


class SnapshotSessionBody(BaseModel):
    machine_name: Optional[str] = None


# You can only snapshot a new machine
@router.post("/session/{session_id}/snapshot")
async def snapshot_session(
    request: Request,
    session_id: str,
    body: Optional[SnapshotSessionBody] = None,
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    gpuEvent = cast(
        Optional[GPUEvent],
        (
            await db.execute(
                (
                    select(GPUEvent)
                    .where(GPUEvent.session_id == session_id)
                    .where(GPUEvent.end_time.is_(None))
                    .apply_org_check(request)
                )
            )
        ).scalar_one_or_none(),
    )

    if gpuEvent is None:
        raise HTTPException(status_code=404, detail="GPUEvent not found")

    modal_function_id = gpuEvent.modal_function_id

    if modal_function_id is None:
        raise HTTPException(status_code=400, detail="Modal function id not found")

    if not modal_function_id.startswith("sb-"):
        raise HTTPException(
            status_code=400, detail="Modal function id is not a sandbox"
        )

    machine = None

    if (
        gpuEvent.machine_id is None
        and body is not None
        and body.machine_name is not None
    ):
        machine = Machine(
            id=uuid4(),
            type=MachineType.COMFY_DEPLOY_SERVERLESS,
            user_id=user_id,
            org_id=org_id,
            status="ready",
            endpoint="not-ready",
            created_at=func.now(),
            updated_at=func.now(),
            name=body.machine_name,
            gpu=gpuEvent.gpu,
            machine_builder_version="4",
        )
        db.add(machine)
        await db.flush()

        gpuEvent.machine_id = machine.id

    if gpuEvent.machine_id is None:
        raise HTTPException(status_code=400, detail="Machine id not found")

    # Get the sandbox and create snapshot
    sb = await modal.Sandbox.from_id.aio(modal_function_id)
    image = sb.snapshot_filesystem()
    image_id = image.object_id

    print("image_id", image_id, image)

    # Get current user info
    current_user = request.state.current_user
    user_id = current_user["user_id"]

    if machine is None:
        machine = await db.execute(
            select(Machine)
            .where(Machine.id == gpuEvent.machine_id)
            .apply_org_check(request)
        )
        machine = machine.scalar_one()

    # Get next version number
    current_version = await db.execute(
        select(func.max(MachineVersion.version)).where(
            MachineVersion.machine_id == machine.id
        )
    )
    next_version = (current_version.scalar() or 0) + 1

    # Create new version with image_id
    machine_version = MachineVersion(
        id=uuid4(),
        machine_id=machine.id,
        version=next_version,
        user_id=user_id,
        created_at=func.now(),
        updated_at=func.now(),
        modal_image_id=image_id,
        **{col: getattr(machine, col) for col in get_machine_columns().keys()},
    )
    db.add(machine_version)

    # Update machine with new version id
    machine.machine_version_id = machine_version.id
    machine.updated_at = func.now()

    await db.commit()
    await db.refresh(machine)
    await db.refresh(gpuEvent)
    await db.refresh(machine_version)

    await redeploy_machine(request, db, background_tasks, machine, machine_version)

    return JSONResponse(
        content={
            "message": "Session snapshot created successfully",
            "version_id": str(machine_version.id),
            "version": next_version,
            "image_id": image_id,
        }
    )


class DeleteSessionResponse(BaseModel):
    success: bool


# Delete a session by id
@router.delete(
    "/session/{session_id}",
    openapi_extra={
        "x-speakeasy-name-override": "cancel",
    },
)
async def delete_session(
    request: Request, session_id: str, db: AsyncSession = Depends(get_db)
) -> DeleteSessionResponse:
    gpuEvent = cast(
        Optional[GPUEvent],
        (
            await db.execute(
                (
                    select(GPUEvent)
                    .where(GPUEvent.session_id == session_id)
                    .where(GPUEvent.end_time.is_(None))
                    .apply_org_check(request)
                )
            )
        ).scalar_one_or_none(),
    )

    if gpuEvent is None:
        raise HTTPException(status_code=404, detail="GPUEvent not found")

    modal_function_id = gpuEvent.modal_function_id
    if modal_function_id is None:
        raise HTTPException(status_code=400, detail="Modal function id not found")

    if modal_function_id.startswith("sb-"):
        await modal.Sandbox.from_id(modal_function_id).terminate.aio()
    else:
        await modal.functions.FunctionCall.from_id(modal_function_id).cancel.aio()

    return {"success": True}
