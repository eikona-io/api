import asyncio
import os
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import joinedload

from .utils import ensure_run_timeout, get_user_settings, post_process_outputs, select

# from sqlalchemy import select
from api.models import (
    GPUEvent,
    MachineVersion,
    WorkflowRun,
    Machine,
    WorkflowRunWithExtra,
    Workflow,
)
from api.database import get_db_context, get_db
from typing import Optional, cast, Any
from uuid import UUID
import logging
import json
from typing import Optional
from uuid import UUID
import datetime as dt
import uuid
from upstash_redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis_asyncio

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Log"])

# Initialize Redis client for log streaming (REST API) only if env is present
_redis_url_log = os.getenv("UPSTASH_REDIS_REST_URL_LOG")
_redis_token_log = os.getenv("UPSTASH_REDIS_REST_TOKEN_LOG")
redis = (
    Redis(url=_redis_url_log, token=_redis_token_log)
    if _redis_url_log and _redis_token_log
    else None
)

# Initialize Redis client for pub/sub using TCP connection
def get_redis_pubsub_client():
    """Get Redis client for pub/sub operations using TCP Redis URL."""
    redis_tcp_url = os.getenv("REDIS_URL_REALTIME")
    if not redis_tcp_url:
        raise ValueError("REDIS_URL_REALTIME environment variable not set")
    
    return redis_asyncio.from_url(redis_tcp_url)

# Initialize Redis client for blocking stream reads (logs)
_redis_tcp_url_log = os.getenv("REDIS_URL_LOG")
redis_stream_client = (
    redis_asyncio.from_url(_redis_tcp_url_log) if _redis_tcp_url_log else None
)

# Sentinel helper to end a stream immediately
async def signal_stream_end(run_id: str):
    try:
        if redis_stream_client is not None:
            # Use TCP client when available
            await redis_stream_client.xadd(run_id, {"message": "__END__"})
        elif redis is not None:
            # Fallback to REST execute
            await redis.execute(["XADD", run_id, "*", "message", "__END__"])
    except Exception as e:
        logger.warning(f"Failed to signal stream end for {run_id}: {e}")

# Initialize consumer group helper
consumer_group = None

# Active streams are now tracked in Redis with keys like "active_stream:<run_id>"

def get_consumer_group():
    global consumer_group
    if consumer_group is None:
        from ..utils.redis_consumer_group import RedisStreamConsumerGroup
        if redis is None:
            raise ValueError("Redis is not configured for log streaming")
        consumer_group = RedisStreamConsumerGroup(redis)
    return consumer_group


# ==================== ACTIVE STREAM MANAGEMENT ====================

async def register_active_stream(run_id: str, client_id: str = "default"):
    """Register an active stream in Redis with TTL."""
    try:
        if redis is None:
            return
        stream_key = f"active_stream:{run_id}"
        stream_data = {
            "client_id": client_id,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "active"
        }
        # Set with 2-hour TTL to auto-cleanup stale entries
        await redis.set(stream_key, json.dumps(stream_data), ex=7200)
        logger.debug(f"Registered active stream for run {run_id}, client {client_id}")
    except Exception as e:
        logger.warning(f"Failed to register active stream for {run_id}: {e}")


async def unregister_active_stream(run_id: str):
    """Remove active stream registration from Redis."""
    try:
        if redis is None:
            return
        stream_key = f"active_stream:{run_id}"
        await redis.execute(["DEL", stream_key])
        logger.debug(f"Unregistered active stream for run {run_id}")
    except Exception as e:
        logger.warning(f"Failed to unregister active stream for {run_id}: {e}")


"""
Note: We previously supported cancellation via Redis keys. With sentinel-based
stream termination, we no longer need cancellation checks/updates. The helpers
above are intentionally removed to avoid extra GET/SET traffic.
"""


def _to_iso_z(ts: dt.datetime) -> str:
    """Format a timezone-aware datetime to ISO 8601 with milliseconds and 'Z'."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    ts_utc = ts.astimezone(dt.timezone.utc)
    try:
        return ts_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except TypeError:
        # For older Python versions without timespec support
        return ts_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@router.get("/v2/stream-logs")
async def stream_logs_endpoint_smart(
    request: Request,
    run_id: Optional[str] = None,
    machine_id_version: Optional[str] = None,
    session_id: Optional[str] = None,
    log_level: Optional[str] = None,
    client_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    target_id = run_id or machine_id_version or session_id
    
    if run_id:
        run_query = select(WorkflowRun).where(WorkflowRun.id == run_id)
        run_result = await db.execute(run_query)
        run = run_result.scalar_one_or_none()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Check if the workflow exists
        workflow_exists_query = select(
            (
                select(1)
                .where(Workflow.id == run.workflow_id)
                .where(~Workflow.deleted)
                .apply_org_check_by_type(Workflow, request)
            ).exists()
        )
        workflow_exists = await db.scalar(workflow_exists_query)

        if not workflow_exists:
            raise HTTPException(
                status_code=403, detail="Not authorized to access these logs"
            )
    elif machine_id_version:
        machine_version_query = select(MachineVersion).where(MachineVersion.id == machine_id_version)
        machine_version_result = await db.execute(machine_version_query)
        machine_version = machine_version_result.scalar_one_or_none()
        
        if not machine_version:
            raise HTTPException(status_code=404, detail="Machine version not found")
        
        machine_exist_query = select(Machine).where(Machine.id == machine_version.machine_id).apply_org_check(request)
        machine_exist_result = await db.execute(machine_exist_query)
        machine_exist = machine_exist_result.scalar_one_or_none()

        if not machine_exist:
            raise HTTPException(status_code=404, detail="Machine not found")

    elif session_id:
        session_query = select(GPUEvent).where(GPUEvent.session_id == session_id).apply_org_check(request)
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
    """
    Smart log streaming that checks for archived logs first.
    If archived logs exist, streams from cache. Otherwise, streams live.
    """
    # Generate unique client ID if not provided
    if client_id is None:
        client_id = f"client_{uuid.uuid4().hex[:8]}"
    
    # First check if logs are archived
    archived_logs = await get_archived_logs(target_id)
    
    if archived_logs:
        # Stream from archived logs
        return StreamingResponse(
            stream_archived_logs(archived_logs, log_level),
            media_type="text/event-stream",
        )
    else:
        # Stream live logs using blocking reads over TCP Redis
        return StreamingResponse(
            stream_logs_blocking(target_id, log_level, client_id),
            media_type="text/event-stream",
        )


async def stream_logs_blocking(run_id: str, log_level: Optional[str] = None, client_id: str = "default"):
    """
    Stream logs using TCP Redis with blocking XREAD. No polling loop, minimal idle commands.
    Each client tails the stream independently (broadcast semantics).
    """
    await register_active_stream(run_id, client_id)

    if redis_stream_client is None:
        # Fallback if misconfigured
        yield f"data: {json.dumps({'type': 'stream_unavailable'})}\n\n"
        return

    stream_name = run_id
    # Start from new messages only
    last_id = "$"

    try:
        while True:
            try:
                # Block indefinitely until new entries arrive, one at a time
                entries = await redis_stream_client.xread(
                    streams={stream_name: last_id},
                    count=1,
                    block=0,
                )
            except Exception as e:
                logger.warning(f"xread error for {run_id}: {e}")
                await asyncio.sleep(1)
                continue

            if not entries:
                continue

            for stream, items in entries:
                for message_id, fields in items:
                    try:
                        # redis-py returns dict of field->value
                        if isinstance(fields, dict):
                            value = fields.get(b"message") or fields.get("message")
                        else:
                            # Defensive: support tuple/list format [key, value]
                            value = fields[1] if len(fields) >= 2 else None

                        if value is None:
                            continue

                        if isinstance(value, bytes):
                            value = value.decode("utf-8")

                        # End-of-stream sentinel
                        if value == "__END__":
                            yield f"data: {json.dumps({'type': 'stream_complete', 'source': 'sentinel'})}\n\n"
                            return

                        try:
                            log_data = json.loads(value)
                        except json.JSONDecodeError:
                            log_data = value

                        for log_entry in normalize_log_data(log_data, log_level):
                            yield f"data: {json.dumps(log_entry)}\n\n"

                        # Advance tail position
                        last_id = message_id
                    except Exception as e:
                        logger.error(f"Error processing Redis stream entry: {e}")
                        continue
    except asyncio.CancelledError:
        logger.info(f"Client {client_id} stream cancelled for run {run_id}")
        raise
    except Exception as e:
        logger.error(f"Error in blocking Redis stream for client {client_id}: {e}")
        raise
    finally:
        await unregister_active_stream(run_id)


def normalize_log_data(log_data: Any, log_level_filter: Optional[str] = None) -> list:
    """
    Normalize different log data formats to the expected schema.
    Returns a list of log entries in the format:
    {"message": message, "level": level, "timestamp": _to_iso_z(timestamp)}
    """
    normalized_logs = []
    current_time = dt.datetime.now(dt.timezone.utc)
    
    if isinstance(log_data, list):
        # Handle array of log entries (body.logs format)
        for entry in log_data:
            if isinstance(entry, dict):
                # Expected format: {"timestamp": unix_timestamp, "logs": "message"}
                message = entry.get("logs", str(entry))
                level = entry.get("level", "info")
                
                # Parse timestamp
                if "timestamp" in entry:
                    try:
                        timestamp = dt.datetime.fromtimestamp(entry["timestamp"], tz=dt.timezone.utc)
                    except (ValueError, TypeError):
                        timestamp = current_time
                else:
                    timestamp = current_time
                
                # Apply log level filter if specified
                if log_level_filter is None or level == log_level_filter:
                    normalized_logs.append({
                        "message": message,
                        "level": level,
                        "timestamp": _to_iso_z(timestamp)
                    })
    elif isinstance(log_data, dict):
        # Handle single log entry or ws_event format
        if "logs" in log_data and "timestamp" in log_data:
            # Single log entry format
            message = log_data["logs"]
            level = "info"
            try:
                timestamp = dt.datetime.fromtimestamp(log_data["timestamp"], tz=dt.timezone.utc)
            except (ValueError, TypeError):
                timestamp = current_time
        else:
            # ws_event or other dict format - convert to string
            message = json.dumps(log_data)
            level = "info"
            timestamp = current_time
        
        # Apply log level filter if specified
        if log_level_filter is None or level == log_level_filter:
            normalized_logs.append({
                "message": message,
                "level": level,
                "timestamp": _to_iso_z(timestamp)
            })
    elif isinstance(log_data, str):
        # Handle string data
        # Apply log level filter if specified
        if log_level_filter is None or "info" == log_level_filter:
            normalized_logs.append({
                "message": log_data,
                "level": "info",
                "timestamp": _to_iso_z(current_time)
            })
    else:
        # Handle any other data type
        # Apply log level filter if specified
        if log_level_filter is None or "info" == log_level_filter:
            normalized_logs.append({
                "message": str(log_data),
                "level": "info",
                "timestamp": _to_iso_z(current_time)
            })
    
    return normalized_logs


# @router.get("/stream-logs")
# async def stream_logs_endpoint(
#     request: Request,
#     run_id: Optional[str] = None,
#     workflow_id: Optional[str] = None,
#     session_id: Optional[str] = None,
#     machine_id: Optional[str] = None,
#     log_level: Optional[str] = None,
#     # db: AsyncSession = Depends(get_db),
#     client=Depends(get_clickhouse_client),
# ):
#     if sum(bool(x) for x in [run_id, workflow_id, machine_id, session_id]) != 1:
#         raise HTTPException(
#             status_code=400, detail="Exactly one ID type must be provided"
#         )

#     id_type = "run" if run_id else "workflow" if workflow_id else "machine"
#     id_value = run_id or workflow_id or machine_id or session_id
    
#     if session_id:
#         id_type = "session"
        
#     print(f"id_type: {id_type}, id_value: {id_value}")

#     return StreamingResponse(
#         stream_logs(id_type, id_value, request, client, log_level),
#         media_type="text/event-stream",
#     )


async def stream_logs(
    id_type: str, id_value: str, request: Request, client, log_level: Optional[str]
):
    try:
        # Get the current user from the request state
        current_user = request.state.current_user
        is_from_session = id_type == "session"

        async with get_db_context() as db:
            # Verify the entity exists and check permissions
            if id_type == "session":
                event = await db.execute(
                select(GPUEvent)
                .where(GPUEvent.session_id == id_value)
                .apply_org_check(request)
            )
                event = event.scalars().first()
                if not event:
                    raise HTTPException(status_code=404, detail="Session not found")
                entity = event
                id_type = "run"
            else:
                model = {"run": WorkflowRun, "workflow": Workflow, "machine": Machine}.get(
                    id_type
                )
                if not model:
                    raise HTTPException(status_code=400, detail="Invalid ID type")

                entity_query = select(model).where(model.id == id_value)
                result = await db.execute(entity_query)
                entity = result.scalar_one_or_none()

                if not entity:
                    raise HTTPException(
                        status_code=404, detail=f"{id_type.capitalize()} not found"
                    )

            # Check permissions based on workflow access for runs
            if id_type == "run" and not is_from_session:
                # Get the workflow associated with this run
                workflow_query = (
                    select(Workflow)
                    .where(Workflow.id == entity.workflow_id)
                    .where(~Workflow.deleted)
                    .apply_org_check(request)
                )
                workflow_result = await db.execute(workflow_query)
                workflow = workflow_result.scalar_one_or_none()
                
                if not workflow:
                    raise HTTPException(
                        status_code=403, detail="Not authorized to access these logs"
                    )
            else:
                # Original permission check for workflow and machine
                has_permission = False
                if hasattr(entity, "org_id") and entity.org_id is not None:
                    has_permission = entity.org_id == current_user.get("org_id")
                elif hasattr(entity, "user_id") and entity.user_id is not None:
                    has_permission = (
                        entity.user_id == current_user.get("user_id")
                        and current_user.get("org_id") is None
                    )

                if not has_permission:
                    raise HTTPException(
                        status_code=403, detail="Not authorized to access these logs"
                    )

        # Stream logs
        last_timestamp = None
        while True:
            query = f"""
            SELECT timestamp, log_level, message
            FROM log_entries
            WHERE {id_type}_id = '{id_value}'
            {f"AND timestamp > '{last_timestamp}'" if last_timestamp else ""}
            {f"AND log_level = '{log_level}'" if log_level else ""}
            ORDER BY timestamp ASC
            LIMIT 100
            """
            result = await client.query(query)
            for row in result.result_rows:
                timestamp, level, message = row
                last_timestamp = timestamp
                yield f"data: {json.dumps({'message': message, 'level': level, 'timestamp': _to_iso_z(timestamp)})}\n\n"

            if not result.result_rows:
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

            await asyncio.sleep(1)  # Wait for 1 second before next query

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        raise
    # finally:
    #     await client.close()  # Ensure the client is closed


# @router.get("/stream-progress")
# async def stream_progress_endpoint(
#     request: Request,
#     run_id: Optional[str] = None,
#     workflow_id: Optional[str] = None,
#     machine_id: Optional[str] = None,
#     return_run: Optional[bool] = False,
#     from_start: Optional[bool] = False,

#     # filter params
#     status: Optional[str] = None,
#     deployment_id: Optional[str] = None,

#     client=Depends(get_clickhouse_client),
# ):
#     if sum(bool(x) for x in [run_id, workflow_id, machine_id]) != 1:
#         raise HTTPException(
#             status_code=400, detail="Exactly one ID type must be provided"
#         )

#     id_type = "run" if run_id else "workflow" if workflow_id else "machine"
#     id_value = run_id or workflow_id or machine_id

#     return StreamingResponse(
#         stream_progress(id_type, id_value, request, client, return_run, from_start, status, deployment_id),
#         media_type="text/event-stream",
#     )


async def stream_progress(
    id_type: str,
    id_value: str,
    request: Request,
    client,
    return_run: Optional[bool] = False,
    from_start: Optional[bool] = False,

    # filter params
    status: Optional[str] = None,
    deployment_id: Optional[str] = None,
):
    async with get_db_context() as db:
        user_settings = await get_user_settings(request, db)

    try:
        # Get the current user from the request state
        current_user = request.state.current_user

        # Verify the entity exists and check permissions
        model = {"run": WorkflowRun, "workflow": Workflow, "machine": Machine}.get(
            id_type
        )
        if not model:
            raise HTTPException(status_code=400, detail="Invalid ID type")

        async with get_db_context() as db:
            entity_query = select(model).where(model.id == id_value)
            result = await db.execute(entity_query)
            entity = result.scalar_one_or_none()

            if not entity:
                raise HTTPException(
                    status_code=404, detail=f"{id_type.capitalize()} not found"
                )

        # Check permissions based on org_id or user_id
        has_permission = False
        if hasattr(entity, "org_id") and entity.org_id is not None:
            has_permission = entity.org_id == current_user.get("org_id")
        elif hasattr(entity, "user_id") and entity.user_id is not None:
            has_permission = (
                entity.user_id == current_user.get("user_id")
                and current_user.get("org_id") is None
            )

        if not has_permission:
            raise HTTPException(
                status_code=403, detail="Not auxhorized to access this progress"
            )

        # Stream progress updates from ClickHouse
        last_update_time = None if from_start else dt.datetime.now(dt.timezone.utc)
        while True:
            # Convert last_update_time to the format expected by ClickHouse
            formatted_time = (
                last_update_time.strftime("%Y-%m-%d %H:%M:%S.%f")
                if last_update_time is not None
                else None
            )

            query = f"""
            SELECT run_id, workflow_id, workflow_version_id, machine_id, timestamp, progress, log, log_type
            FROM workflow_events
            WHERE {id_type}_id = '{id_value}'
            {f"AND timestamp > toDateTime64('{formatted_time}', 6)" if formatted_time is not None else ""}
            ORDER BY timestamp ASC
            LIMIT 100
            """
            result = await client.query(query)

            if result.result_rows and return_run:
                async with get_db_context() as db:
                    # get a set of unique run ids
                    run_ids = set()
                    for row in result.result_rows:
                        run_ids.add(row[0])

                    # update the last_update_time to the timestamp of the last row
                    if result.result_rows:
                        last_update_time = result.result_rows[-1][4]

                    for run_id in run_ids:
                        query = (
                            select(WorkflowRunWithExtra)
                            .options(joinedload(WorkflowRun.outputs))
                            .where(WorkflowRun.id == run_id)
                            # .apply_org_check(request)
                        )

                        if status:
                            query = query.where(WorkflowRun.status == status)

                        if deployment_id:
                            query = query.where(WorkflowRun.deployment_id == deployment_id)

                        result = await db.execute(query)
                        run = result.unique().scalar_one_or_none()

                        if not run:
                            # raise HTTPException(status_code=404, detail="Run not found")
                            continue

                        run = cast(WorkflowRun, run)
                        ensure_run_timeout(run)
                        await post_process_outputs(run.outputs, user_settings)
                        # Convert the run to a dictionary and remove the run_log
                        # run_dict = {k: v for k, v in vars(run).items() if k != "run_log"}
                        run_dict = run.to_dict()
                        run_dict.pop("run_log", None)
                        yield f"data: {json.dumps(run_dict)}\n\n"
            elif result.result_rows:
                for row in result.result_rows:
                    (
                        run_id,
                        workflow_id,
                        workflow_version_id,
                        machine_id,
                        timestamp,
                        progress,
                        log,
                        status,
                    ) = row
                    last_update_time = timestamp
                    progress_data = {
                        "run_id": str(run_id),
                        "workflow_id": str(workflow_id),
                        "machine_id": str(machine_id),
                        "progress": progress,
                        "status": status,
                        "node_class": log,
                        "timestamp": timestamp.isoformat()[:-3] + "Z",
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"

                    if status in ["success", "failed", "timeout", "cancelled"]:
                        return  # Exit the function if the final status is reached
            else:
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

            await asyncio.sleep(1)  # Wait for 1 second before next query

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        raise

# @router.get("/clickhouse-run-logs/{run_id}")
# async def get_clickhouse_run_logs(
#     run_id: UUID,
#     request: Request,
#     client=Depends(get_clickhouse_client),
# ):
#     org_id = request.state.current_user.get("org_id", None)
#     user_id = request.state.current_user.get("user_id")

#     query = """
#         SELECT run_id, timestamp, log, user_id, org_id, log_type
#         FROM workflow_events
#         WHERE run_id = %(run_id)s
#         AND log_type != 'input'
#         AND log != ''
#         AND (
#             (%(org_id)s IS NOT NULL AND org_id = %(org_id)s)
#             OR (%(org_id)s IS NULL AND org_id IS NULL AND user_id = %(user_id)s)
#         )
#         ORDER BY timestamp ASC
#     """

#     result = await client.query(
#         query,
#         parameters={
#             "run_id": str(run_id),
#             "org_id": str(org_id) if org_id else None,
#             "user_id": str(user_id)
#         }
#     )

#     # Convert rows to list of dicts with properly formatted timestamps
#     formatted_rows = [
#         {
#             "run_id": str(row[0]),
#             "timestamp": row[1].isoformat(),
#             "log": row[2],
#             "user_id": str(row[3]) if row[3] else None,
#             "org_id": str(row[4]) if row[4] else None,
#             "log_type": row[5]
#         }
#         for row in result.result_rows
#     ]

#     return formatted_rows


@router.get("/v2/clickhouse-run-logs/{run_id}")
async def get_run_logs_v2(
    run_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    v2 version that fetches logs from Redis instead of Clickhouse.
    Returns archived logs if available, otherwise returns empty list.
    """
    # Validate run exists and user has access
    run_query = (
        select(WorkflowRun.created_at)
        .where(WorkflowRun.id == run_id)
        .apply_org_check(request)
        .limit(1)
    )
    run_result = await db.execute(run_query)
    run_created_at = run_result.scalar_one_or_none()
    if not run_created_at:
        raise HTTPException(status_code=404, detail="Run not found")

    # Get archived logs from Redis
    archived_logs = await get_archived_logs(str(run_id))
    
    if not archived_logs:
        # Return empty list if no logs found in Redis
        return []
    
    # Since logs are stored per run_id in Redis and we already validated
    # user access to this run, we only need to filter by log type/content
    filtered_logs = []
    for log_entry in archived_logs:
        # Filter out input logs and empty logs (same as v1)
        if (
            # log_entry.get("log_type") != "input"
            log_entry.get("message", "").strip() != ""
        ):
            # Ensure consistent format with v1 endpoint
            formatted_entry = {
                "run_id": str(log_entry.get("run_id", run_id)),
                "timestamp": log_entry.get("timestamp"),
                "log": log_entry.get("message"),
                "user_id": log_entry.get("user_id"),
                "org_id": log_entry.get("org_id"),
                "log_type": log_entry.get("log_type"),
            }
            filtered_logs.append(formatted_entry)
    
    # Sort by timestamp (same as v1)
    filtered_logs.sort(key=lambda x: x.get("timestamp", ""))
    
    return filtered_logs


# ==================== LOG ARCHIVAL FUNCTIONS ====================

async def archive_logs_for_run(run_id: str):
    """
    Archive all logs for a completed run to Redis with 30-day TTL.
    This should be called when a run reaches a terminal state.
    """
    try:
        if redis is None:
            return 0
        # Collect all logs from the stream
        stream_name = run_id
        archived_logs = []
        
        # Read all entries from the stream (from beginning)
        entries = await redis.execute(["XREAD", "STREAMS", stream_name, "0"])
        
        if entries and len(entries) > 0:
            # Process each stream entry
            for stream, items in entries:
                for message_id, fields in items:
                    if len(fields) >= 2:
                        try:
                            # Parse the serialized value
                            serialized_value = fields[1]
                            if isinstance(serialized_value, bytes):
                                serialized_value = serialized_value.decode('utf-8')
                            
                            # Try to parse as JSON, fallback to string
                            try:
                                log_data = json.loads(serialized_value)
                            except json.JSONDecodeError:
                                log_data = serialized_value
                            
                            # Normalize the data to the expected schema
                            normalized_logs = normalize_log_data(log_data)
                            archived_logs.extend(normalized_logs)
                            
                        except Exception as e:
                            logger.error(f"Error processing log entry for archival: {e}")
                            continue
        
        if archived_logs:
            # Store archived logs with 30-day TTL (2592000 seconds)
            archive_key = f"log:{run_id}"
            await redis.set(archive_key, json.dumps(archived_logs), ex=2592000)
            logger.info(f"Archived {len(archived_logs)} log entries for run {run_id}")
            
            # Clean up the stream after archiving
            await cleanup_stream(run_id)
            
        return len(archived_logs)
        
    except Exception as e:
        logger.error(f"Error archiving logs for run {run_id}: {e}")
        return 0


async def get_archived_logs(run_id: str):
    """
    Retrieve archived logs for a run from Redis.
    Returns None if no archived logs exist.
    """
    try:
        if redis is None:
            return None
        archive_key = f"log:{run_id}"
        archived_data = await redis.get(archive_key)
        
        if archived_data:
            if isinstance(archived_data, bytes):
                archived_data = archived_data.decode('utf-8')
            return json.loads(archived_data)
        
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving archived logs for run {run_id}: {e}")
        return None


async def stream_archived_logs(archived_logs: list, log_level: Optional[str] = None):
    """
    Simulate streaming from archived logs.
    """
    try:
        for log_entry in archived_logs:
            # Apply log level filter if specified
            if log_level and log_entry.get("level", "").lower() != log_level.lower():
                continue
                
            yield f"data: {json.dumps(log_entry)}\n\n"
            # Small delay to simulate streaming
            # await asyncio.sleep(0.01)
            
        # Send completion signal
        yield f"data: {json.dumps({'type': 'stream_complete', 'source': 'archive'})}\n\n"
        
    except Exception as e:
        logger.error(f"Error streaming archived logs: {e}")
        yield f"data: {json.dumps({'error': 'Error streaming archived logs'})}\n\n"


async def cleanup_stream(run_id: str):
    """
    Clean up the Redis stream and related data after archiving.
    """
    try:
        stream_name = run_id
        
        # Delete the stream
        await redis.execute(["DEL", stream_name])
        
        # Clean up per-run offset key if still used (avoid KEYS/SCAN)
        try:
            await redis.execute(["DEL", f"last_stream_id:{run_id}"])
        except Exception as e:
            logger.debug(f"No last_stream_id for {run_id} to delete: {e}")
        
        # Clean up active stream key
        await unregister_active_stream(run_id)
        
        logger.info(f"Cleaned up stream and related data for run {run_id}")
        
    except Exception as e:
        logger.error(f"Error cleaning up stream for run {run_id}: {e}")


async def cancel_active_streams(run_id: str):
    """
    Backward-compat shim. With sentinel-based termination we simply signal end.
    """
    await signal_stream_end(run_id)


@router.get("/v2/stream-progress")
async def stream_progress_endpoint_v2(
    request: Request,
    run_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    machine_id: Optional[str] = None,
    return_run: Optional[bool] = False,
    
    # filter params
    status: Optional[str] = None,
    deployment_id: Optional[str] = None,
    
    db: AsyncSession = Depends(get_db),
):
    """
    V2 stream progress endpoint that uses Redis pub/sub for real-time updates
    instead of polling ClickHouse. Much more cost-effective with Upstash.
    """
    if sum(bool(x) for x in [run_id, workflow_id, machine_id]) != 1:
        raise HTTPException(
            status_code=400, detail="Exactly one ID type must be provided"
        )

    id_type = "run" if run_id else "workflow" if workflow_id else "machine"
    id_value = run_id or workflow_id or machine_id

    # Verify permissions similar to the original endpoint
    try:
        current_user = request.state.current_user
        model = {"run": WorkflowRun, "workflow": Workflow, "machine": Machine}.get(id_type)
        if not model:
            raise HTTPException(status_code=400, detail="Invalid ID type")

        entity_query = select(model).where(model.id == id_value)
        result = await db.execute(entity_query)
        entity = result.scalar_one_or_none()

        if not entity:
            raise HTTPException(
                status_code=404, detail=f"{id_type.capitalize()} not found"
            )

        # Check permissions
        has_permission = False
        if hasattr(entity, "org_id") and entity.org_id is not None:
            has_permission = entity.org_id == current_user.get("org_id")
        elif hasattr(entity, "user_id") and entity.user_id is not None:
            has_permission = (
                entity.user_id == current_user.get("user_id")
                and current_user.get("org_id") is None
            )

        if not has_permission:
            raise HTTPException(
                status_code=403, detail="Not authorized to access this progress"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Permission check failed: {str(e)}")

    return StreamingResponse(
        stream_progress_v2(id_type, id_value, request, return_run, status, deployment_id),
        media_type="text/event-stream",
    )


async def stream_progress_v2(
    id_type: str,
    id_value: str,
    request: Request,
    return_run: Optional[bool] = False,
    status: Optional[str] = None,
    deployment_id: Optional[str] = None,
):
    """
    V2 stream progress method that uses Redis pub/sub for real-time updates.
    Subscribes to Redis channels instead of polling ClickHouse.
    """
    try:
        # Determine the appropriate Redis channel(s) to subscribe to.
        # Use scoped channels for run streams to reduce per-connection traffic.
        # if id_type == "run":
        # Frontend not using per-run streams → prefer user/org scoped channels
        current_user = request.state.current_user
        org_id = current_user.get("org_id")
        user_id = current_user.get("user_id")
        channels = [f"progress:org:{org_id}"] if org_id else [f"progress:user:{user_id}"]
        # else:
        #     # For workflow/machine aggregated views you can keep the global
        #     # channel if needed. If your frontend always has org/user context,
        #     # you can also scope these similarly. For now, keep global.
        #     channels = ["workflow_events"]
        
        # Subscribe to the Redis channels
        redis_client = None
        pubsub = None
        try:
            # Create Redis pub/sub client
            redis_client = get_redis_pubsub_client()
            pubsub = redis_client.pubsub()
            
            logger.info(f"Starting Redis pub/sub stream for {id_type} {id_value}, channels: {channels}")
            
            # Subscribe to the appropriate channels
            for channel in channels:
                await pubsub.subscribe(channel)
                logger.info(f"Subscribed to Redis channel: {channel}")
            
            # Send initial connection confirmation
            yield f"data: {json.dumps({'type': 'connection_established', 'channels': channels})}\n\n"
            
            # Listen for messages
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        # Parse the message data
                        message_data = json.loads(message['data'])

                        # Support compact short-key payloads as well as legacy payloads
                        # r=run_id, w=workflow_id, m=machine_id, s=status, p=progress, t=timestamp, n=node/log
                        run_id_msg = message_data.get("run_id") or message_data.get("r")
                        workflow_id_msg = message_data.get("workflow_id") or message_data.get("w")
                        machine_id_msg = message_data.get("machine_id") or message_data.get("m")
                        status_msg = message_data.get("status") or message_data.get("s")
                        progress_msg = message_data.get("progress") if message_data.get("progress") is not None else message_data.get("p", 0)
                        timestamp_msg = message_data.get("timestamp") or message_data.get("t")
                        node_msg = message_data.get("log") or message_data.get("n", "")
                        
                        # Filter messages based on the request parameters
                        should_include = True
                        
                        if id_type == "workflow" and workflow_id_msg != id_value:
                            should_include = False
                        elif id_type == "machine" and machine_id_msg != id_value:
                            should_include = False
                        
                        # Apply status filter if specified
                        if status and status_msg != status:
                            should_include = False
                        
                        if should_include:
                            if return_run:
                                # Fetch the full run data when requested
                                try:
                                    async with get_db_context() as db:
                                        # Get the run_id from the message data
                                        message_run_id = run_id_msg
                                        if not message_run_id:
                                            logger.warning("No run_id in message data for returnRun request")
                                            continue
                                        
                                        run_query = (
                                            select(WorkflowRunWithExtra)
                                            .options(joinedload(WorkflowRun.outputs))
                                            .where(WorkflowRun.id == message_run_id)
                                        )
                                        
                                        # Apply additional filters based on the original request
                                        if id_type == "workflow":
                                            run_query = run_query.where(WorkflowRun.workflow_id == id_value)
                                        elif id_type == "machine":
                                            run_query = run_query.where(WorkflowRun.machine_id == id_value)
                                        
                                        if status:
                                            run_query = run_query.where(WorkflowRun.status == status)
                                        
                                        if deployment_id:
                                            run_query = run_query.where(WorkflowRun.deployment_id == deployment_id)
                                        
                                        result = await db.execute(run_query)
                                        run = result.unique().scalar_one_or_none()
                                        
                                        if run:
                                            run = cast(WorkflowRun, run)
                                            ensure_run_timeout(run)
                                            user_settings = await get_user_settings(request, db)
                                            await post_process_outputs(run.outputs, user_settings)
                                            
                                            run_dict = run.to_dict()
                                            run_dict.pop("run_log", None)
                                            logger.debug(f"Sending full run object for run {message_run_id}, keys: {list(run_dict.keys())}")
                                            yield f"data: {json.dumps(run_dict)}\n\n"
                                        else:
                                            logger.warning(f"Run {message_run_id} not found in database with query filters")
                                except Exception as e:
                                    logger.error(f"Error fetching run data: {e}")
                                    continue
                            else:
                                # Send the progress data directly (normalized fields)
                                progress_data = {
                                    "run_id": run_id_msg,
                                    "workflow_id": workflow_id_msg,
                                    "machine_id": machine_id_msg,
                                    "progress": progress_msg,
                                    "status": status_msg,
                                    "node_class": node_msg,
                                    "timestamp": timestamp_msg,
                                }
                                logger.debug(f"Sending progress data for run {message_data.get('run_id')}")
                                yield f"data: {json.dumps(progress_data)}\n\n"
                            
                            # Check for terminal status - only break for run-specific streams
                            if (id_type == "run" and 
                                (status_msg) in ["success", "failed", "timeout", "cancelled"]):
                                logger.info(f"Terminal status received for run stream: {message_data.get('status')}")
                                break
                                
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing Redis message: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                        continue
                
                elif message['type'] == 'subscribe':
                    logger.info(f"Successfully subscribed to channel: {message['channel']}")
                    # Send confirmation that we're connected and listening
                    yield f"data: {json.dumps({'type': 'subscribed', 'channel': message['channel'].decode()})}\n\n"
                    
        except asyncio.CancelledError:
            logger.info(f"Redis pub/sub stream cancelled for {id_type} {id_value}")
            raise
        except Exception as e:
            logger.error(f"Error in Redis pub/sub stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                    
        finally:
            # Clean up the pub/sub connection
            if pubsub:
                try:
                    await pubsub.unsubscribe(*channels)
                    await pubsub.close()
                except Exception as e:
                    logger.warning(f"Error cleaning up pub/sub connection: {e}")
            
            if redis_client:
                try:
                    await redis_client.close()
                except Exception as e:
                    logger.warning(f"Error closing Redis client: {e}")
            
    except Exception as e:
        logger.error(f"Error in stream_progress_v2: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


def is_terminal_status(status: str) -> bool:
    """
    Check if a status is a terminal state.
    """
    terminal_statuses = ["success", "failed", "timeout", "cancelled"]
    return status in terminal_statuses
