import asyncio
from datetime import timedelta
import hashlib
import json

# from http.client import HTTPException
from fastapi import Request, HTTPException, Depends
import os
from uuid import UUID
import uuid

from api.modal.builder import (
    BuildMachineItem,
    GPUType,
    KeepWarmBody,
    build_logic,
    set_machine_always_on,
)
from api.routes.volumes import retrieve_model_volumes
from api.utils.docker import (
    DockerCommandResponse,
    generate_all_docker_commands,
    comfyui_hash,
)
from pydantic import BaseModel
from .types import (
    GPUEventModel,
    MachineGPU,
    MachineModel,
    MachineType,
)
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from .utils import generate_persistent_token, select, is_valid_uuid
from sqlalchemy import func, cast, String, or_
from fastapi.responses import JSONResponse

from api.models import (
    Deployment,
    GPUEvent,
    Machine,
    Workflow,
    MachineVersion,
    get_machine_columns,
)

# from sqlalchemy import select
from api.database import get_db
import logging
from typing import Any, Dict, List, Optional, Literal
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
    search: Optional[str] = None,
    limit: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    machines_query = (
        select(Machine)
        .order_by(Machine.created_at.desc())
        .where(~Machine.deleted)
        .apply_org_check(request)
    )

    if search:
        if is_valid_uuid(search):
            # Exact UUID match - most efficient
            machines_query = machines_query.where(Machine.id == search)
        else:
            # Name search using trigram similarity for better performance
            machines_query = machines_query.where(Machine.name.ilike(f"%{search}%"))

    if limit:
        machines_query = machines_query.limit(limit)

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


class DockerCommand(BaseModel):
    when: Literal["before", "after"]
    commands: List[str]


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
    extra_docker_commands: Optional[List[DockerCommand]] = None
    machine_builder_version: Optional[str] = "4"
    base_docker_image: Optional[str] = None
    python_version: Optional[str] = None
    extra_args: Optional[str] = None
    prestart_command: Optional[str] = None
    keep_warm: Optional[int] = 0


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
    extra_docker_commands: Optional[List[DockerCommand]] = None
    machine_builder_version: Optional[str] = None
    base_docker_image: Optional[str] = None
    python_version: Optional[str] = None
    extra_args: Optional[str] = None
    prestart_command: Optional[str] = None
    keep_warm: Optional[int] = None
    is_trigger_rebuild: Optional[bool] = False


current_endpoint = os.getenv("CURRENT_API_URL")


async def create_machine_version(
    db: AsyncSession,
    machine: Machine,
    user_id: str,
    version: int = 1,
    current_version_data: MachineVersion = None,
) -> MachineVersion:
    # if the machine builder version is not 4, pass this function
    if machine.machine_builder_version != "4":
        return

    new_version_id = uuid.uuid4()

    # Create new version without transaction (caller manages transaction)
    machine_version = MachineVersion(
        id=new_version_id,
        machine_id=machine.id,
        version=version,
        user_id=user_id,
        created_at=func.now(),
        updated_at=func.now(),
        **{col: getattr(machine, col) for col in get_machine_columns().keys()},
        modal_image_id=current_version_data.modal_image_id
        if current_version_data
        else None,
    )
    db.add(machine_version)
    await db.flush()

    # Update machine with new version
    machine.machine_version_id = machine_version.id
    machine.updated_at = func.now()

    return machine_version


def hash_machine_dependencies(docker_commands: DockerCommandResponse):
    return hashlib.sha256(json.dumps(docker_commands.model_dump()).encode()).hexdigest()


@router.post("/machine/serverless")
async def create_serverless_machine(
    request: Request,
    machine: ServerlessMachineModel,
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> MachineModel:
    new_machine_id = uuid.uuid4()
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    docker_commands = generate_all_docker_commands(machine)
    docker_commands_hash = hash_machine_dependencies(docker_commands)

    async with db.begin():  # Single transaction for entire operation
        # Create initial machine
        machine = Machine(
            **machine.model_dump(),
            id=new_machine_id,
            type=MachineType.COMFY_DEPLOY_SERVERLESS,
            user_id=user_id,
            org_id=org_id,
            status="building",
            endpoint="not-ready",
            created_at=func.now(),
            updated_at=func.now(),
            machine_hash=docker_commands_hash,
        )
        db.add(machine)
        await db.flush()

        # Create initial version (uses same transaction)
        await create_machine_version(db, machine, user_id)

    # Transaction automatically commits here if successful
    await db.refresh(machine)  # Only one refresh at the end

    volumes = await retrieve_model_volumes(request, db)
    # docker_commands = generate_all_docker_commands(machine)
    machine_token = generate_persistent_token(user_id, org_id)

    if machine.machine_hash is not None:
        existing_machine_info_url = f"https://comfyui.comfydeploy.com/static-assets/{machine.machine_hash}/object_info.json"
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.head(existing_machine_info_url) as response:
                    skip_static_assets = response.status == 200
        except Exception as e:
            logger.warning(f"Error checking static assets: {e}")
            skip_static_assets = False
    else:
        skip_static_assets = False

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
        # skip_static_assets=skip_static_assets,
        skip_static_assets=True,
        docker_commands=docker_commands.model_dump()["docker_commands"],
        machine_builder_version=machine.machine_builder_version,
        base_docker_image=machine.base_docker_image,
        python_version=machine.python_version,
        prestart_command=machine.prestart_command,
        extra_args=machine.extra_args,
        machine_version_id=str(machine.machine_version_id),
        machine_hash=docker_commands_hash,
    )

    background_tasks.add_task(build_logic, params)

    return JSONResponse(content=machine.to_dict())


@router.patch("/machine/serverless/{machine_id}")
async def update_serverless_machine(
    request: Request,
    machine_id: UUID,
    update_machine: UpdateServerlessMachineModel,
    rollback_version_id: Optional[UUID] = None,
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

    if machine.machine_version_id is None and machine.machine_builder_version == "4":
        # give it a default version 1
        await create_machine_version(db, machine, user_id)
        await db.commit()
        await db.refresh(machine)  # Single refresh at the end

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

    rebuild = update_machine_dict.get(
        "is_trigger_rebuild", False
    ) or check_fields_for_changes(
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

    docker_commands = generate_all_docker_commands(machine)
    docker_commands_hash = hash_machine_dependencies(docker_commands)
    machine.machine_hash = docker_commands_hash

    machine.updated_at = func.now()

    if rebuild:
        machine.status = "building"

        # Get next version number
        if not rollback_version_id:
            current_version = await db.execute(
                select(func.max(MachineVersion.version)).where(
                    MachineVersion.machine_id == machine.id
                )
            )
            next_version = (current_version.scalar() or 0) + 1

            current_version_data = await db.execute(
                select(MachineVersion).where(
                    MachineVersion.machine_id == machine.id
                    and MachineVersion.version == current_version.scalar()
                )
            )
            current_version_data = current_version_data.scalars().first()

            print("current_version_data", current_version_data)

            # Create new version
            machine_version = await create_machine_version(
                db,
                machine,
                user_id,
                version=next_version,
                current_version_data=current_version_data,
            )
        else:
            machine.machine_version_id = rollback_version_id

            # update that machine version's created_at to now
            machine_version = await db.execute(
                select(MachineVersion).where(MachineVersion.id == rollback_version_id)
            )
            machine_version = machine_version.scalars().first()
            machine_version.created_at = func.now()
            machine_version.updated_at = func.now()
            machine_version.status = machine.status
            await db.commit()

        if machine.machine_hash is not None:
            existing_machine_info_url = f"https://comfyui.comfydeploy.com/static-assets/{machine.machine_hash}/object_info.json"
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.head(existing_machine_info_url) as response:
                        skip_static_assets = response.status == 200
            except Exception as e:
                logger.warning(f"Error checking static assets: {e}")
                skip_static_assets = False
        else:
            skip_static_assets = False

        # Prepare build parameters
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
            # skip_static_assets=skip_static_assets,
            skip_static_assets=True,
            docker_commands=docker_commands.model_dump()["docker_commands"],
            machine_builder_version=machine.machine_builder_version,
            base_docker_image=machine.base_docker_image,
            python_version=machine.python_version,
            prestart_command=machine.prestart_command,
            extra_args=machine.extra_args,
            machine_version_id=str(machine.machine_version_id),
            machine_hash=docker_commands_hash,
            modal_image_id=machine_version.modal_image_id,
        )
        background_tasks.add_task(build_logic, params)

    if keep_warm_changed and not rebuild:
        print("Keep warm changed", machine.keep_warm)
        set_machine_always_on(
            str(machine.id),
            KeepWarmBody(warm_pool_size=machine.keep_warm, gpu=GPUType(machine.gpu)),
        )

    await db.commit()
    await db.refresh(machine)

    return JSONResponse(content=machine.to_dict())


async def redeploy_machine(
    request: Request,
    db: AsyncSession,
    background_tasks: BackgroundTasks,
    machine: Machine,
    machine_version: MachineVersion,
):
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    volumes = await retrieve_model_volumes(request, db)
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
        # skip_static_assets=skip_static_assets,
        skip_static_assets=True,
        modal_image_id=machine_version.modal_image_id,
        machine_builder_version=machine.machine_builder_version,
        base_docker_image=machine.base_docker_image,
        python_version=machine.python_version,
        prestart_command=machine.prestart_command,
        extra_args=machine.extra_args,
        machine_version_id=str(machine.machine_version_id),
        machine_hash=machine_version.machine_hash,
    )
    print("params", params)
    background_tasks.add_task(build_logic, params)


@router.get("/machine/serverless/{machine_id}/versions")
async def get_machine_versions(
    request: Request,
    machine_id: UUID,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    # First check if user has access to this machine
    await get_machine(request, machine_id, db)

    machine_versions = await db.execute(
        select(MachineVersion)
        .where(MachineVersion.machine_id == machine_id)
        .order_by(MachineVersion.version.desc())
        .paginate(limit, offset)
    )

    return JSONResponse(
        content=[version.to_dict() for version in machine_versions.scalars().all()]
    )


@router.get("/machine/serverless/{machine_id}/versions/all")
async def get_all_machine_versions(
    request: Request,
    machine_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    # First check if user has access to this machine
    await get_machine(request, machine_id, db)

    machine_versions = await db.execute(
        select(MachineVersion)
        .where(MachineVersion.machine_id == machine_id)
        .order_by(MachineVersion.version.desc())
    )
    return JSONResponse(
        content=[version.to_dict() for version in machine_versions.scalars().all()]
    )


# get specific machine version
@router.get("/machine/serverless/{machine_id}/versions/{version_id}")
async def get_machine_version(
    request: Request,
    machine_id: UUID,
    version_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    # First check if user has access to this machine
    await get_machine(request, machine_id, db)

    machine_version = await db.execute(
        select(MachineVersion).where(MachineVersion.id == version_id)
    )
    machine_version = machine_version.scalars().first()
    return JSONResponse(content=machine_version.to_dict())


class RollbackMachineVersionBody(BaseModel):
    machine_version_id: Optional[UUID] = None
    version: Optional[int] = None


@router.post("/machine/serverless/{machine_id}/rollback")
async def rollback_serverless_machine(
    request: Request,
    machine_id: UUID,
    version: RollbackMachineVersionBody,
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    async def get_machine(machine_id: UUID):
        machine = await db.execute(
            select(Machine).where(Machine.id == machine_id).apply_org_check(request)
        )
        machine = machine.scalars().first()
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")
        return machine

    if version.machine_version_id is None and version.version is None:
        raise HTTPException(
            status_code=400,
            detail="Either machine_version_id or version must be provided",
        )

    machine = await get_machine(machine_id)

    if version.machine_version_id is not None:
        if machine.machine_version_id == version.machine_version_id:
            raise HTTPException(
                status_code=400, detail="Cannot rollback to current version"
            )

        machine_version = await db.execute(
            select(MachineVersion).where(
                MachineVersion.id == version.machine_version_id
            )
        )
        machine_version = machine_version.scalars().first()
    else:
        # Get current version first
        current_version = await db.execute(
            select(MachineVersion).where(
                MachineVersion.id == machine.machine_version_id
            )
        )
        current_version = current_version.scalars().first()

        if current_version and current_version.version == version.version:
            raise HTTPException(
                status_code=400, detail="Cannot rollback to current version"
            )

        # Get target version
        machine_version = await db.execute(
            select(MachineVersion)
            .where(MachineVersion.machine_id == machine_id)
            .where(MachineVersion.version == version.version)
        )
        machine_version = machine_version.scalars().first()

    if machine_version is None:
        raise HTTPException(status_code=404, detail="Machine version not found")

    # Call update_serverless_machine with rollback flag
    return await update_serverless_machine(
        request=request,
        machine_id=machine_id,
        update_machine=UpdateServerlessMachineModel(
            **{
                col: getattr(machine_version, col)
                for col in get_machine_columns().keys()
                if hasattr(machine_version, col)
            }
        ),
        rollback_version_id=machine_version.id,
        db=db,
        background_tasks=background_tasks,
    )


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
        id=uuid.uuid4(),
        name=machine.name,
        endpoint=machine.endpoint,
        type=machine.type,
        auth_token=machine.auth_token,
        user_id=user_id,
        org_id=org_id,
        created_at=func.now(),
        updated_at=func.now(),
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
