import asyncio
from datetime import datetime
import os
from pprint import pprint
from api.routes.types import GPUEventModel, MachineGPU
from api.utils.docker import (
    CustomNode,
    DepsBody,
    DockerStep,
    DockerSteps,
    comfyui_cmd,
    generate_all_docker_commands,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import modal
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import (
    select,
)
from pydantic import BaseModel, Field

# from sqlalchemy import select
from api.models import (
    GPUEvent,
    Machine,
)
from api.database import get_db, get_db_context
from typing import Any, Dict, List, Optional, cast, Union
from uuid import UUID, uuid4
import logging
from typing import Optional
from sqlalchemy import update
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Session"])
beta_router = APIRouter(tags=["Beta"])

status_endpoint = os.environ.get("CURRENT_API_URL") + "/api/update-run"


def get_comfy_runner(machine_id: str, session_id: str | UUID, timeout: int, gpu: str):
    logger.info(machine_id)
    ComfyDeployRunner = modal.Cls.lookup(str(machine_id), "ComfyDeployRunner")
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
    url: str


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
    request: Request, machine_id: str, db: AsyncSession = Depends(get_db)
) -> List[GPUEventModel]:
    result = await db.execute(
        (
            select(GPUEvent)
            .where(GPUEvent.machine_id == machine_id)
            .where(GPUEvent.end_time.is_(None))
            .where(GPUEvent.session_id.isnot(None))
            .apply_org_check(request)
        )
    )
    return result.scalars().all()

async def create_session_background_task(
    machine_id: str,
    session_id: UUID,
    request: Request,
    timeout: int,
    gpu: str,
    status_queue: Optional[asyncio.Queue] = None,
):
    runner = get_comfy_runner(machine_id, session_id, timeout, gpu)
    print("async_creation", status_queue)
    async with modal.Queue.ephemeral() as q:
        result = await runner.create_tunnel.spawn.aio(q, status_endpoint)
        modal_function_id = result.object_id

        gpuEvent = None
        while gpuEvent is None:
            async with get_db_context() as db:
                gpuEvent = cast(
                    Optional[GPUEvent],
                    (
                        await db.execute(
                            (
                                select(GPUEvent)
                                .where(GPUEvent.session_id == str(session_id))
                                .where(GPUEvent.end_time.is_(None))
                                .apply_org_check(request)
                            )
                        )
                    ).scalar_one_or_none(),
                )

            if gpuEvent is None:
                await asyncio.sleep(1)

        print("async_creation", status_queue)
        if status_queue is not None:
            await status_queue.put(gpuEvent.modal_function_id)
        print("async_creation", gpuEvent)

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
    machine_id: str
    gpu: MachineGPU = Field(None, description="The GPU to use")
    timeout: Optional[int] = Field(None, description="The timeout in minutes")
    dependencies: Optional[Union[List[str], DepsBody]] = Field(
        None,
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
    session_id: UUID,
    body: CreateDynamicSessionBody,
    status_queue: Optional[asyncio.Queue] = None,
):
    app = modal.App(str(session_id))
    
    gpu_event_id = uuid4()

    try:
        async with app.run.aio():
            async with get_db_context() as db:
                # Insert GPU event
                new_gpu_event = GPUEvent(
                    id=gpu_event_id,
                    session_id=str(session_id),
                    machine_id=body.machine_id,
                    gpu=body.gpu.value if body.gpu is not None else "CPU",
                    session_timeout=body.timeout or 15,
                    gpu_provider="modal",
                    start_time=datetime.now(),
                    # Add other necessary fields here
                )
                db.add(new_gpu_event)
                await db.commit()
                await db.refresh(new_gpu_event)
            
            dockerfile_image = modal.Image.debian_slim(python_version="3.11")

            if body.dependencies:
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
                    converted = generate_all_docker_commands(deps_body)
                else:
                    converted = generate_all_docker_commands(body.dependencies)

                pprint(converted)

                # return {
                #     "session_id": session_id,
                #     "url": "",
                # }

                docker_commands = converted.docker_commands
                if docker_commands is not None:
                    for commands in docker_commands:
                        dockerfile_image = dockerfile_image.dockerfile_commands(
                            commands,
                        )

            sb = await modal.Sandbox.create.aio(
                "bash",
                "-c",
                comfyui_cmd(cpu=True if body.gpu == "CPU" else False),
                image=dockerfile_image,
                timeout=(body.timeout or 15) * 60,
                gpu=body.gpu.value
                if body.gpu is not None and body.gpu.value != "CPU"
                else None,
                app=app,
                workdir="/comfyui",
                encrypted_ports=[8188],
            )

            logger.info(sb.tunnels())
            tunnel = sb.tunnels()[8188]

            await status_queue.put(tunnel.url)

            logger.info(tunnel.url)

            await sb.wait.aio()
    except Exception as e:
        async with get_db_context() as db:
            new_gpu_event.end_time = datetime.now()
            new_gpu_event.error = str(e)
            await db.commit()
            await db.refresh(new_gpu_event)
            
        raise e


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
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    session_id = uuid4()
    q = asyncio.Queue()

    task = asyncio.create_task(
        create_dynamic_sesssion_background_task(session_id, body, q)
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

    modal.functions.FunctionCall.from_id(modal_function_id).cancel()

    return {"success": True}
