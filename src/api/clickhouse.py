from uuid import uuid4
from datetime import datetime
from typing import List, Tuple

async def insert_to_clickhouse(client, table_name: str, data: List[Tuple]):
    """
    Function to insert data into ClickHouse table.

    :param client: ClickHouse client
    :param table_name: Name of the table to insert data into
    :param data: List of tuples containing the data to be inserted
    """
    query = f"INSERT INTO {table_name} VALUES"
    await client.execute(query, data)


async def insert_workflow_event(
    client,
    user_id: str,
    org_id: str | None,
    machine_id: str,
    gpu_event_id: str,
    workflow_id: str,
    workflow_version_id: str,
    run_id: str,
    log_type: str,
    progress: float = 0.0,
    log: str = ""
):
    """
    Function to handle inserting data into workflow_events table in ClickHouse.

    :param client: ClickHouse client
    :param user_id: ID of the user
    :param org_id: ID of the organization (can be None)
    :param machine_id: ID of the machine
    :param gpu_event_id: ID of the GPU event (can be None)
    :param workflow_id: ID of the workflow
    :param workflow_version_id: ID of the workflow version (can be None)
    :param run_id: ID of the run
    :param log_type: Type of the log event
    :param log: Log data
    """
    timestamp = datetime.utcnow()
    event_data = [
        (
            user_id,
            org_id if org_id else None,
            machine_id,
            gpu_event_id if gpu_event_id else None,
            workflow_id,
            workflow_version_id if workflow_version_id else None,
            run_id,
            timestamp,
            log_type,
            progress,
            log
        )
    ]
    await insert_to_clickhouse(client, "workflow_events", event_data)


# NOTE: not used
async def insert_progress_update(client, run_id: str, workflow_id: str, machine_id: str, progress: int, live_status: str, status: str):
    """
    Function to handle inserting data into progress_updates table in ClickHouse.

    :param client: ClickHouse client
    :param run_id: ID of the run
    :param workflow_id: ID of the workflow
    :param machine_id: ID of the machine
    :param progress: Progress value
    :param live_status: Live status value
    :param status: Status value
    """
    updated_at = datetime.utcnow()
    progress_data = [
        (
            uuid4(),
            run_id,
            workflow_id,
            machine_id,
            updated_at,
            progress,
            live_status,
            status,
        )
    ]
    await insert_to_clickhouse(client, "progress_updates", progress_data)
