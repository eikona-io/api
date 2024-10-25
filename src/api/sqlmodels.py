from datetime import datetime
import enum
from typing import List, Optional, Dict
from pydantic import BaseModel
from sqlalchemy import JSON, Column, MetaData
from sqlmodel import SQLModel, Field
import uuid

class WorkflowRunStatus(str, enum.Enum):
    NOT_STARTED = "not-started"
    RUNNING = "running"
    UPLOADING = "uploading"
    SUCCESS = "success"
    FAILED = "failed"
    STARTED = "started"
    QUEUED = "queued"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


metadata = MetaData(schema="comfyui_deploy")


class WorkflowRunOutput(SQLModel, table=True):
    __tablename__ = "workflow_run_outputs"
    metadata = metadata

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    run_id: uuid.UUID = Field(foreign_key="workflow_runs.id")
    data: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    node_meta: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field()
    updated_at: datetime = Field()


# class WorkflowRunWebhookBody(BaseModel):
#     status: WorkflowRunStatus
#     live_status: Optional[str]
#     progress: float
#     run_id: str
#     outputs: List[WorkflowRunOutput]


class WorkflowRunWebhookResponse(BaseModel):
    status: str
