import asyncio
import os
from fastapi import APIRouter, Depends, HTTPException, Request
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
from typing import Optional, cast
from uuid import UUID, uuid4
import logging
from typing import Optional
from sqlalchemy import update
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Session"])

status_endpoint = os.environ.get("CURRENT_API_URL") + "/api/update-run"


def get_comfy_runner(machine_id: str, session_id: str | UUID, timeout: int, gpu: str):
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


# Return the session tunnel url
@router.get("/session/{session_id}")
async def get_session(
    request: Request, session_id: str, db: AsyncSession = Depends(get_db)
):
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


# Return the sessions for a machine
@router.get("/machine/{machine_id}/sessions")
async def get_machine_sessions(
    request: Request, machine_id: str, db: AsyncSession = Depends(get_db)
):
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
            url = await q.get.aio()

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


class CreateSessionBody(BaseModel):
    gpu: Optional[str] = Field(None, description="The GPU to use")
    timeout: Optional[int] = Field(None, description="The timeout in minutes")
    async_creation: bool = Field(
        False, description="Whether to create the session asynchronously"
    )

async def ensure_session_creation_complete(task: asyncio.Task):
    try:
        await task
    except Exception as e:
        # Handle any exceptions that occurred during session creation
        logger.error(f"Session creation failed: {str(e)}")
        # You might want to update the session status in the database here

# Create a new session for a machine, return the session id and url
@router.post("/machine/{machine_id}/session")
async def create_session(
    request: Request,
    body: CreateSessionBody,
    background_tasks: BackgroundTasks,
    machine_id: str,
    db: AsyncSession = Depends(get_db),
):
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
    if body.async_creation:
        print("async_creation")
        q = asyncio.Queue()
        
        task = asyncio.create_task(
            create_session_background_task(
                machine_id,
                session_id,
                request,
                body.timeout or 15,
                body.gpu if body.gpu is not None else machine.gpu,
                q
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
        await create_session_background_task(
            db,
            machine_id,
            session_id,
            request,
            body.timeout or 15,
            body.gpu if body.gpu is not None else machine.gpu,
        )

    return {
        "session_id": session_id,
    }


# Delete a session by id
@router.delete("/session/{session_id}")
async def delete_session(
    request: Request, session_id: str, db: AsyncSession = Depends(get_db)
):
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
