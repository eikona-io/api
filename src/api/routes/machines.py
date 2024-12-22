import asyncio
from datetime import timedelta
# from http.client import HTTPException
from fastapi import Request, HTTPException, Depends
import os
from uuid import UUID
import uuid

from api.modal.builder import (
    BuildMachineItem,
    KeepWarmBody,
    build_logic,
    set_machine_always_on,
)
from api.routes.volumes import get_model_volumes, retrieve_model_volumes
from api.utils.docker import generate_all_docker_commands, comfyui_hash
from pydantic import BaseModel
from .types import (
    GPUEventModel,
    MachineGPU,
    MachineModel,
    MachineType,
    WorkflowVersionModel,
)
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from .utils import generate_persistent_token, select
from sqlalchemy import func
from fastapi.responses import JSONResponse

from api.models import Deployment, GPUEvent, Machine, Workflow

# from sqlalchemy import select
from api.database import get_db
import logging
from typing import Any, Dict, List, Optional, Union
# from fastapi_pagination import Page, add_pagination, paginate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Machine"])
public_router = APIRouter(tags=["Machine"])


@router.get("/machines", response_model=List[MachineModel])
async def get_machines(
    request: Request,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    is_deleted: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
):
    machines_query = (
        select(Machine)
        .order_by(Machine.created_at.desc())
        .where(Machine.deleted == is_deleted)
        .apply_org_check(request)
        .paginate(limit, offset)
    )

    if search:
        machines_query = machines_query.where(
            func.lower(Machine.name).contains(search.lower())
        )

    result = await db.execute(machines_query)
    machines = result.unique().scalars().all()

    if not machines:
        # raise HTTPException(status_code=404, detail="Runs not found")
        return []

    machines_data = [machine.to_dict() for machine in machines]

    return JSONResponse(content=machines_data)


@router.get("/machines/all", response_model=List[MachineModel])
async def get_all_machines(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    machines_query = (
        select(Machine)
        .order_by(Machine.created_at.desc())
        .where(~Machine.deleted)
        .apply_org_check(request)
    )

    result = await db.execute(machines_query)
    machines = result.unique().scalars().all()

    return JSONResponse(content=[machine.to_dict() for machine in machines])


@router.get("/machine/{machine_id}", response_model=MachineModel)
async def get_machine(
    request: Request,
    machine_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    machine = await db.execute(
        select(Machine)
        .where(Machine.id == machine_id)
        .where(~Machine.deleted)
        .apply_org_check(request)
    )
    machine = machine.scalars().first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    return JSONResponse(content=machine.to_dict())


@router.get("/machine/{machine_id}/events", response_model=List[GPUEventModel])
async def get_machine_events(
    request: Request,
    machine_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    # Calculate the timestamp for 24 hours ago
    twenty_four_hours_ago = func.now() - timedelta(hours=24)

    events = await db.execute(
        select(GPUEvent)
        .where(GPUEvent.machine_id == machine_id)
        .where(GPUEvent.start_time >= twenty_four_hours_ago)
        .order_by(GPUEvent.start_time.desc())
        .apply_org_check(request)
    )
    events = events.scalars().all()
    return JSONResponse(content=[event.to_dict() for event in events])


class ServerlessMachineModel(BaseModel):
    name: str
    comfyui_version: Optional[str] = comfyui_hash
    gpu: MachineGPU
    docker_command_steps: Optional[Dict[str, Any]] = {"steps": []}
    allow_concurrent_inputs: int = 1
    concurrency_limit: int = 2
    install_custom_node_with_gpu: bool = False
    # ws_timeout: int = 2
    run_timeout: int = 300
    idle_timeout: int = 60
    extra_docker_commands: Optional[Dict[str, Any]] = None
    machine_builder_version: Optional[str] = "4"
    base_docker_image: Optional[str] = None
    python_version: Optional[str] = None
    extra_args: Optional[str] = None
    prestart_command: Optional[str] = None


class UpdateServerlessMachineModel(BaseModel):
    name: Optional[str] = None
    comfyui_version: Optional[str] = None
    gpu: Optional[MachineGPU] = None
    docker_command_steps: Optional[Dict[str, Any]] = None
    allow_concurrent_inputs: Optional[int] = None
    concurrency_limit: Optional[int] = None
    install_custom_node_with_gpu: Optional[bool] = None
    run_timeout: Optional[int] = None
    idle_timeout: Optional[int] = None
    extra_docker_commands: Optional[Dict[str, Any]] = None
    machine_builder_version: Optional[str] = None
    base_docker_image: Optional[str] = None
    python_version: Optional[str] = None
    extra_args: Optional[str] = None
    prestart_command: Optional[str] = None
    keep_warm: Optional[int] = None


current_endpoint = os.getenv("CURRENT_API_URL")


@router.post("/machine/serverless")
async def create_serverless_machine(
    request: Request,
    machine: ServerlessMachineModel,
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> MachineModel:
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    machine = Machine(
        **machine.model_dump(),
        id=uuid.uuid4(),
        type=MachineType.COMFY_DEPLOY_SERVERLESS,
        user_id=user_id,
        org_id=org_id,
        status="building",
        endpoint="not-ready",
        created_at=func.now(),
        updated_at=func.now(),
    )

    volumes = await retrieve_model_volumes(request, db)
    docker_commands = generate_all_docker_commands(machine)
    machine_token = generate_persistent_token(user_id, org_id)

    db.add(machine)
    await db.commit()
    await db.refresh(machine)

    params = BuildMachineItem(
        machine_id=str(machine.id),
        name=str(machine.id),
        cd_callback_url=f"{current_endpoint}/api/machine-built",
        callback_url=f"{current_endpoint}/api",
        gpu_event_callback_url=f"{current_endpoint}/api/gpu_event",
        models=machine.models,
        gpu=machine.gpu,
        model_volume_name=volumes[0]["volume_name"],
        run_timeout=machine.run_timeout,
        idle_timeout=machine.idle_timeout,
        auth_token=machine_token,
        ws_timeout=machine.ws_timeout,
        concurrency_limit=machine.concurrency_limit,
        allow_concurrent_inputs=machine.allow_concurrent_inputs,
        legacy_mode=machine.legacy_mode,
        install_custom_node_with_gpu=machine.install_custom_node_with_gpu,
        allow_background_volume_commits=machine.allow_background_volume_commits,
        retrieve_static_assets=machine.retrieve_static_assets,
        skip_static_assets=False,
        docker_commands=docker_commands.model_dump()["docker_commands"],
        machine_builder_version=machine.machine_builder_version,
        base_docker_image=machine.base_docker_image,
        python_version=machine.python_version,
        prestart_command=machine.prestart_command,
        extra_args=machine.extra_args,
    )

    background_tasks.add_task(build_logic, params)

    return JSONResponse(content=machine.to_dict())


@router.patch("/machine/serverless/{machine_id}")
async def update_serverless_machine(
    request: Request,
    machine_id: UUID,
    update_machine: UpdateServerlessMachineModel,
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> MachineModel:
    machine = await db.execute(
        select(Machine).where(Machine.id == machine_id).apply_org_check(request)
    )
    machine = machine.scalars().first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    if machine.type != MachineType.COMFY_DEPLOY_SERVERLESS:
        raise HTTPException(
            status_code=400, detail="Machine is not a serverless machine"
        )

    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    fields_to_trigger_rebuild = [
        "allow_concurrent_inputs",
        "comfyui_version",
        "concurrency_limit",
        "docker_command_steps",
        "extra_docker_commands",
        "idle_timeout",
        "ws_timeout",
        "machine_builder_version",
        "install_custom_node_with_gpu",
        "run_timeout",
        "base_docker_image",
        "extra_args",
        "prestart_command",
        "python_version",
        "install_custom_node_with_gpu",
    ]

    update_machine_dict = update_machine.model_dump()

    rebuild = check_fields_for_changes(
        machine, update_machine_dict, fields_to_trigger_rebuild
    )
    keep_warm_changed = check_fields_for_changes(
        machine, update_machine_dict, ["keep_warm"]
    )
    gpu_changed = check_fields_for_changes(machine, update_machine_dict, ["gpu"])

    # We dont need to trigger a rebuild if we only change the gpu.
    # We need to trigger a rebuild if we change the gpu and install_custom_node_with_gpu is true
    if gpu_changed and machine.install_custom_node_with_gpu:
        rebuild = True

    print(update_machine.model_dump())

    for key, value in update_machine.model_dump().items():
        if hasattr(machine, key) and value is not None:
            setattr(machine, key, value)

    machine.updated_at = func.now()

    if rebuild:
        volumes = await retrieve_model_volumes(request, db)
        docker_commands = generate_all_docker_commands(machine)
        machine_token = generate_persistent_token(user_id, org_id)
        params = BuildMachineItem(
            machine_id=str(machine.id),
            name=str(machine.id),
            cd_callback_url=f"{current_endpoint}/api/machine-built",
            callback_url=f"{current_endpoint}/api",
            gpu_event_callback_url=f"{current_endpoint}/api/gpu_event",
            models=machine.models,
            gpu=machine.gpu,
            model_volume_name=volumes[0]["volume_name"],
            run_timeout=machine.run_timeout,
            idle_timeout=machine.idle_timeout,
            auth_token=machine_token,
            ws_timeout=machine.ws_timeout,
            concurrency_limit=machine.concurrency_limit,
            allow_concurrent_inputs=machine.allow_concurrent_inputs,
            legacy_mode=machine.legacy_mode,
            install_custom_node_with_gpu=machine.install_custom_node_with_gpu,
            allow_background_volume_commits=machine.allow_background_volume_commits,
            retrieve_static_assets=machine.retrieve_static_assets,
            skip_static_assets=False,
            docker_commands=docker_commands.model_dump()["docker_commands"],
            machine_builder_version=machine.machine_builder_version,
            base_docker_image=machine.base_docker_image,
            python_version=machine.python_version,
            prestart_command=machine.prestart_command,
            extra_args=machine.extra_args,
        )
        machine.status = "building"
        background_tasks.add_task(build_logic, params)

    if keep_warm_changed and not rebuild:
        print("Keep warm changed", machine.keep_warm)
        set_machine_always_on(
            machine.id, KeepWarmBody(warm_pool_size=machine.keep_warm, gpu=machine.gpu)
        )

    await db.commit()
    await db.refresh(machine)

    return JSONResponse(content=machine.to_dict())


@router.delete("/machine/{machine_id}")
async def delete_machine(
    request: Request,
    machine_id: UUID,
    force: bool = False,
    db: AsyncSession = Depends(get_db),
):
    machine = await db.execute(
        select(Machine).where(Machine.id == machine_id).apply_org_check(request)
    )
    machine = machine.scalars().first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    if machine.keep_warm > 0:
        raise HTTPException(
            status_code=400,
            detail="Please set keep warm to 0 before deleting the machine.",
        )

    if not force:
        # Check if there are existing deployments
        deployments = await db.execute(
            select(Deployment)
            .join(Workflow)
            .where(Deployment.machine_id == machine_id)
            .with_only_columns(Deployment.workflow_id, Workflow.name)
        )
        deployments = deployments.all()

        if len(deployments) > 0:
            logger.info(f"Deployments: {deployments}")
            workflow_names = ",".join([name for _, name in deployments])
            raise HTTPException(
                status_code=402,
                detail=f"You still have these workflows related to this machine: {workflow_names}",
            )

    machine.deleted = True
    await db.commit()
    await db.refresh(machine)

    if machine.type == MachineType.COMFY_DEPLOY_SERVERLESS and machine.modal_app_id:
        try:
            app_name = machine.modal_app_id
            process = await asyncio.subprocess.create_subprocess_shell(
                "modal app stop " + app_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()
            logger.info(f"Successfully stopped modal app: {app_name}")
        except Exception as e:
            logger.error(f"Error stopping modal app for machine {machine.id}: {str(e)}")

    return JSONResponse(content={"message": "Machine deleted"})


class CustomMachineModel(BaseModel):
    name: str
    type: MachineType
    endpoint: str
    auth_token: Optional[str]


class UpdateCustomMachineModel(BaseModel):
    name: Optional[str]
    type: Optional[MachineType]
    endpoint: Optional[str]
    auth_token: Optional[str]


@router.post("/machine/custom")
async def create_custom_machine(
    request: Request,
    machine: CustomMachineModel,
    db: AsyncSession = Depends(get_db),
) -> MachineModel:
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    machine = Machine(
        name=machine.name,
        endpoint=machine.endpoint,
        type=machine.type,
        auth_token=machine.auth_token,
        user_id=user_id,
        org_id=org_id,
    )

    db.add(machine)
    await db.commit()
    await db.refresh(machine)

    return JSONResponse(content=machine.to_dict())


@router.patch("/machine/custom/{machine_id}")
async def update_custom_machine(
    request: Request,
    machine_id: UUID,
    machine_update: UpdateCustomMachineModel,
    db: AsyncSession = Depends(get_db),
) -> MachineModel:
    machine = await db.execute(
        select(Machine).where(Machine.id == machine_id).apply_org_check(request)
    )
    machine = machine.scalars().first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    for key, value in machine_update.model_dump().items():
        if hasattr(machine, key) and value is not None:
            setattr(machine, key, value)

    machine.updated_at = func.now()

    await db.commit()
    await db.refresh(machine)

    return JSONResponse(content=machine.to_dict())


def has_field_changed(old_obj, new_value, field_name):
    """Check if a field's value has changed."""
    existing_value = getattr(old_obj, field_name)
    return existing_value != new_value


def check_fields_for_changes(old_obj, new_data, fields_to_check):
    """Check if any of the specified fields have changed."""
    for field in fields_to_check:
        if (
            field in new_data
            and new_data[field] is not None
            and hasattr(old_obj, field)
            and has_field_changed(old_obj, new_data[field], field)
        ):
            return True
    return False
