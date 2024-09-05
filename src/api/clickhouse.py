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
