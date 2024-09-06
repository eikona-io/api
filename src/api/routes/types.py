import asyncio
from enum import Enum
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import modal
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from api.routes.internal import send_realtime_update, send_workflow_update
from .utils import select

# from sqlalchemy import select
from api.models import (
    WorkflowRun,
    Deployment,
    Machine,
    WorkflowRunOutput,
    WorkflowVersion,
    Workflow,
)
from api.database import get_db, get_clickhouse_client, get_db_context
from typing import Literal, Optional, Union, cast
from pydantic import BaseModel, Field
from typing import Dict, Any
from uuid import UUID
import logging
from datetime import datetime
import logfire
from pprint import pprint
import json
import io
import httpx
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class WorkflowRunOutputModel(BaseModel):
    id: UUID
    run_id: UUID
    data: Optional[Dict[str, Any]]
    node_meta: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


def format_datetime(dt: Optional[datetime]) -> Optional[str]:
    print(dt)
    if dt is None:
        return None
    return dt.isoformat()[:-3] + "Z"

class WorkflowModel(BaseModel):
    id: UUID
    user_id: str
    org_id: Optional[str]
    name: str
    selected_machine_id: Optional[UUID]
    created_at: datetime = Field(serialization_fn=format_datetime)
    updated_at: datetime = Field(serialization_fn=format_datetime)
    pinned: bool = False
    
    class Config:
        from_attributes = True

class WorkflowListResponse(BaseModel):
    workflows: List[WorkflowModel]
    query_length: int

class WorkflowRunModel(BaseModel):
    id: UUID
    workflow_version_id: Optional[UUID]
    workflow_inputs: Optional[Dict[str, Any]]
    workflow_id: UUID
    workflow_api: Optional[Dict[str, Any]]
    machine_id: Optional[UUID]
    origin: str
    status: str
    ended_at: Optional[datetime] = Field(default=None, serialization_fn=format_datetime)
    created_at: datetime = Field(serialization_fn=format_datetime)
    updated_at: datetime = Field(serialization_fn=format_datetime)
    queued_at: Optional[datetime] = Field(
        default=None, serialization_fn=format_datetime
    )
    started_at: Optional[datetime] = Field(
        default=None, serialization_fn=format_datetime
    )
    gpu_event_id: Optional[str]
    gpu: Optional[str]
    machine_version: Optional[str]
    machine_type: Optional[str]
    modal_function_call_id: Optional[str]
    user_id: Optional[str]
    org_id: Optional[str]
    live_status: Optional[str]
    progress: float = Field(default=0)
    is_realtime: bool = Field(default=False)
    webhook: Optional[str]
    webhook_status: Optional[str]
    webhook_intermediate_status: bool = Field(default=False)
    outputs: List[WorkflowRunOutputModel] = []

    number: int
    # total: int
    duration: Optional[float]
    cold_start_duration: Optional[float]
    cold_start_duration_total: Optional[float]
    run_duration: Optional[float]

    class Config:
        from_attributes = True


class WorkflowRunOrigin(str, Enum):
    MANUAL = "manual"
    API = "api"
    PUBLIC_SHARE = "public-share"


class WorkflowRequestShare(BaseModel):
    execution_mode: Optional[Literal["async", "sync", "sync_first_result"]] = "async"
    inputs: Dict[str, Any] = Field(default_factory=dict)

    webhook: Optional[str] = None
    webhook_intermediate_status: Optional[bool] = False

    origin: Optional[str] = "api"
    batch_number: Optional[int] = None

    batch_input_params: Optional[Dict[str, List[Any]]] = Field(
        default=None,
        example={
            "input_number": [1, 2, 3],
            "input_text": ["apple", "banana", "cherry"],
        },
        description="Optional dictionary of batch input parameters. Keys are input names, values are lists of inputs.",
    )


class WorkflowRunRequest(WorkflowRequestShare):
    workflow_id: UUID
    workflow_api_json: str
    machine_id: Optional[UUID] = None


class WorkflowRunVersionRequest(WorkflowRequestShare):
    workflow_version_id: UUID
    machine_id: Optional[UUID] = None


class DeploymentRunRequest(WorkflowRequestShare):
    deployment_id: UUID


CreateRunRequest = Union[
    WorkflowRunVersionRequest, WorkflowRunRequest, DeploymentRunRequest
]


class CreateRunResponse(BaseModel):
    run_id: UUID


class Input(BaseModel):
    prompt_id: str
    workflow_api: Optional[dict] = None
    inputs: Optional[dict]
    workflow_api_raw: dict
    status_endpoint: str
    file_upload_endpoint: str


class WorkflowRunOutputModel(BaseModel):
    id: UUID
    run_id: UUID
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    type: Optional[str] = None
    node_id: Optional[str] = None

    class Config:
        from_attributes = True
