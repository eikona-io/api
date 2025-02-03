from enum import Enum

from api.sqlmodels import WorkflowRunStatus


# from sqlalchemy import select
from typing import Annotated, Literal, Optional, Union
from pydantic import ConfigDict, Field, RootModel, WithJsonSchema
from typing import Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import Field
from typing import Optional, List
from pydantic.json_schema import GenerateJsonSchema
from pydantic.json_schema import SkipJsonSchema

# from datetime import datetime
from uuid import UUID

from pydantic import Field
from enum import Enum

from pydantic import BaseModel

# class CustomJsonSchemaGenerator(GenerateJsonSchema):
#     def generate(self, schema, mode='validation'):
#         print("hi")
#         json_schema = super().generate(schema, mode=mode)
#         if 'properties' in json_schema:
#             # Filter out hidden properties
#             json_schema['properties'] = {
#                 k: v for k, v in json_schema['properties'].items()
#                 if not v.get('hidden', False)
#             }
#         return json_schema

# class BaseModel(_BaseModel):
#     model_config = {
#         "schema_generator": CustomJsonSchemaGenerator,
#     }

# class WorkflowRunOutputModel(BaseModel):
#     id: UUID
#     run_id: UUID
#     data: Optional[Dict[str, Any]]
#     node_meta: Optional[Dict[str, Any]]
#     created_at: datetime
#     updated_at: datetime


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
    created_at: datetime = Field()
    updated_at: datetime = Field()
    pinned: bool = False
    deleted: bool = False

    class Config:
        from_attributes = True


class WorkflowListResponse(BaseModel):
    workflows: List[WorkflowModel]
    query_length: int


class MediaItem(BaseModel):
    url: str
    type: str
    filename: str
    is_public: Optional[bool] = None
    subfolder: Optional[str] = None
    upload_duration: Optional[float] = None


class WorkflowRunOutputModel(BaseModel):
    id: UUID
    run_id: UUID
    data: Dict[str, List[Union[MediaItem, str, bool]]]
    node_meta: Optional[Any]
    created_at: datetime
    updated_at: datetime
    type: Optional[str] = None
    node_id: Optional[str] = None

    class Config:
        from_attributes = True


class WorkflowRunWebhookBody(BaseModel):
    run_id: str
    status: WorkflowRunStatus
    live_status: Optional[str]
    progress: float = Field(default=0)
    outputs: List[WorkflowRunOutputModel] = []


class WorkflowRunModel(BaseModel):
    id: UUID
    workflow_version_id: Optional[UUID]
    workflow_inputs: Optional[Any]
    workflow_id: UUID
    workflow_api: Optional[Any]
    machine_id: Optional[UUID]
    origin: str
    status: str
    ended_at: Optional[datetime] = Field(
        default=None,
    )
    created_at: datetime = Field()
    updated_at: datetime = Field()
    queued_at: Optional[datetime] = Field(
        default=None,
    )
    started_at: Optional[datetime] = Field(
        default=None,
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


class WorkflowVersionModel(BaseModel):
    id: UUID
    workflow_id: UUID
    workflow: Dict[str, Any]
    workflow_api: Optional[Dict[str, Any]]
    user_id: Optional[str]
    comment: Optional[str]
    version: int
    snapshot: Optional[Dict[str, Any]]
    dependencies: Optional[Dict[str, Any]]
    created_at: datetime = Field()
    updated_at: datetime = Field()

    class Config:
        from_attributes = True


class WorkflowRunOrigin(str, Enum):
    MANUAL = "manual"
    API = "api"
    PUBLIC_SHARE = "public-share"


class MachineType(str, Enum):
    CLASSIC = "classic"
    RUNPOD_SERVERLESS = "runpod-serverless"
    MODAL_SERVERLESS = "modal-serverless"
    COMFY_DEPLOY_SERVERLESS = "comfy-deploy-serverless"
    WORKSPACE = "workspace"
    WORKSPACE_V2 = "workspace-v2"


class MachineStatus(str, Enum):
    NOT_STARTED = "not-started"
    READY = "ready"
    BUILDING = "building"
    ERROR = "error"
    RUNNING = "running"
    PAUSED = "paused"
    STARTING = "starting"


class MachineGPU(str, Enum):
    T4 = "T4"
    L4 = "L4"
    A10G = "A10G"
    L40S = "L40S"
    A100 = "A100"
    A100_80GB = "A100-80GB"
    H100 = "H100"


class WorkspaceGPU(str, Enum):
    RTX_4090 = "4090"

class DockerCommand(BaseModel):
    when: Literal["before", "after"]
    commands: List[str]

class MachineSharedFields(BaseModel):
    comfyui_version: Optional[str] = None
    gpu: Optional[MachineGPU] = None
    docker_command_steps: Optional[Dict[str, Any]] = None
    allow_concurrent_inputs: int = 1
    concurrency_limit: int = 2
    install_custom_node_with_gpu: bool = False
    run_timeout: int = 300
    idle_timeout: int = 60
    extra_docker_commands: Optional[List[DockerCommand]] = None
    machine_builder_version: Optional[str] = "2"
    base_docker_image: Optional[str] = None
    python_version: Optional[str] = None
    extra_args: Optional[str] = None
    prestart_command: Optional[str] = None
    keep_warm: int = 0

    status: MachineStatus = MachineStatus.READY
    build_log: Optional[str]

class MachineVersionModel(MachineSharedFields):
    id: UUID
    machine_id: UUID
    version: int
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class MachineModel(MachineSharedFields):
    id: UUID
    user_id: str
    name: str
    org_id: Optional[str]
    endpoint: str
    created_at: datetime
    updated_at: datetime
    disabled: bool = False
    auth_token: Optional[str]
    type: MachineType = MachineType.CLASSIC
    static_assets_status: MachineStatus = MachineStatus.NOT_STARTED
    machine_version: Optional[str]
    snapshot: Optional[Dict[str, Any]]
    models: Optional[Dict[str, Any]]
    ws_gpu: Optional[WorkspaceGPU]
    pod_id: Optional[str]
    legacy_mode: bool = False
    ws_timeout: int = 2
    build_machine_instance_id: Optional[str]
    modal_app_id: Optional[str]
    target_workflow_id: Optional[UUID]
    dependencies: Optional[Dict[str, Any]]
    deleted: bool = False
    allow_background_volume_commits: bool = False
    gpu_workspace: bool = False
    retrieve_static_assets: bool = False
    object_info: Optional[Dict[str, Any]]
    object_info_str: Optional[str]
    filename_list_cache: Optional[Dict[str, Any]]
    extensions: Optional[Dict[str, Any]]
    import_failed_logs: Optional[str]
    machine_version_id: Optional[UUID]
    has_workflows: bool = False

    class Config:
        from_attributes = True


class WorkflowRequestShare(BaseModel):
    execution_mode: SkipJsonSchema[
        Optional[Literal["async", "sync", "sync_first_result", "stream"]]
    ] = Field(
        default="async",
        example="async",
    )

    inputs: Dict[str, Union[str, int, float, bool, List[Any]]] = Field(
        # default_factory=dict,
        default={},
        description="The inputs to the workflow",
        example={"prompt": "A beautiful landscape", "seed": 42},
    )
    webhook: Annotated[
        Optional[str], WithJsonSchema(json_schema={"type": "string"})
    ] = Field(
        default=None,
        example="https://example.com/webhook",
        json_schema_extra={"type": "string"},  # This ensures it shows as string in docs
    )
    webhook_intermediate_status: bool = Field(
        default=False,
        example=True,
    )
    origin: SkipJsonSchema[Optional[str]] = Field(
        default="api",
        example="manual",
    )
    batch_number: SkipJsonSchema[Optional[int]] = Field(
        default=None,
        example=5,
    )
    batch_input_params: SkipJsonSchema[Optional[Dict[str, List[Any]]]] = Field(
        default=None,
        example={
            "input_number": [1, 2, 3],
            "input_text": ["apple", "banana", "cherry"],
        },
        description="Optional dictionary of batch input parameters. Keys are input names, values are lists of inputs.",
    )
    is_native_run: SkipJsonSchema[Optional[bool]] = Field(
        default=False,
        example=True,
    )
    gpu_event_id: SkipJsonSchema[Optional[str]] = Field(
        default=None,
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    gpu: Annotated[
        Optional[MachineGPU],
        WithJsonSchema(
            json_schema={"type": "string", "enum": [g.value for g in MachineGPU]}
        ),
    ] = Field(
        default=None,
        description="The GPU to override the machine's default GPU",
    )

    # model_config = {
    #     "json_schema_extra": {
    #         "examples": [
    #             {
    #                 "inputs": {
    #                     "prompt": "A futuristic cityscape",
    #                     "seed": 123456,
    #                     "num_inference_steps": 30,
    #                 },
    #                 "webhook": "https://myapp.com/webhook",
    #             }
    #         ]
    #     }
    # }


class WorkflowRunRequest(WorkflowRequestShare):
    workflow_id: UUID
    workflow_api_json: Dict[str, Any]
    workflow: Annotated[
        Dict[str, Any], WithJsonSchema(json_schema={"type": "object"})
    ] = None
    machine_id: Annotated[
        Optional[UUID], WithJsonSchema(json_schema={"type": "string"})
    ] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "workflow_id": "12345678-1234-5678-1234-567812345678",
                    "workflow_api": {},
                    "inputs": {
                        "prompt": "A futuristic cityscape",
                        "seed": 123456,
                        "num_inference_steps": 30,
                    },
                    "webhook": "https://myapp.com/webhook",
                }
            ]
        }
    }


class WorkflowRunVersionRequest(WorkflowRequestShare):
    workflow_version_id: UUID
    machine_id: Optional[UUID] = None


class DeploymentRunRequest(WorkflowRequestShare):
    deployment_id: UUID = Field(examples=["15e79589-12c9-453c-a41a-348fdd7de957"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "deployment_id": "12345678-1234-5678-1234-567812345678",
                    "inputs": {
                        "prompt": "A futuristic cityscape",
                        "seed": 123456,
                        "num_inference_steps": 30,
                    },
                    "webhook": "https://myapp.com/webhook",
                }
            ]
        }
    }


class ModelRunRequest(WorkflowRequestShare):
    model_id: str


CreateRunRequest = Union[
    WorkflowRunVersionRequest, WorkflowRunRequest, DeploymentRunRequest, ModelRunRequest
]


class CreateRunBatchResponse(BaseModel):
    batch_id: UUID


class CreateRunResponse(BaseModel):
    run_id: UUID = Field(
        ..., description="The ID of the run, use this to get the run status and outputs"
    )


class Input(BaseModel):
    prompt_id: str
    workflow_api: Optional[dict] = None
    inputs: Optional[dict]
    workflow_api_raw: dict
    status_endpoint: str
    file_upload_endpoint: str


class LogDataContent(BaseModel):
    logs: str
    timestamp: datetime = Field(..., description="Timestamp in UTC")


class LogUpdateEvent(BaseModel):
    event: Literal["log_update"] = "log_update"
    data: LogDataContent
    # data: str


class EventUpdate(BaseModel):
    event: Optional[str] = None
    data: Optional[Any] = None


class EventUpdateEvent(BaseModel):
    event: Literal["event_update"] = "event_update"
    # data: str
    data: EventUpdate


# RunStream = Union[LogUpdateEvent, EventUpdateEvent]


# Add this discriminator class
class RunStream(RootModel):
    root: Union[LogUpdateEvent, EventUpdateEvent] = Field(..., discriminator="event")


class WorkflowRunNativeOutputModel(BaseModel):
    prompt_id: str
    workflow_api_raw: Dict[str, Any]
    inputs: Optional[Dict[str, Any]]
    status_endpoint: str
    file_upload_endpoint: str
    cd_token: str
    gpu_event_id: Optional[str] = None

    class Config:
        from_attributes = True


class VolFSStructure(BaseModel):
    contents: List[Union["VolFolder", "VolFile"]]


class VolFolder(BaseModel):
    path: str
    type: Literal["folder"]
    contents: List[Union["VolFolder", "VolFile"]]


class VolFile(BaseModel):
    id: Optional[str] = None
    path: str
    type: Literal["file"]
    size: Optional[float] = None


VolFSStructure.update_forward_refs()


class Model(BaseModel):
    id: UUID
    user_id: str
    org_id: Optional[str]
    description: Optional[str]
    user_volume_id: UUID
    model_name: str
    folder_path: str
    target_symlink_path: str
    civitai_id: Optional[str]
    civitai_version_id: Optional[str]
    civitai_url: Optional[str]
    civitai_download_url: Optional[str]
    civitai_model_response: Optional[Dict[str, Any]]
    hf_url: Optional[str]
    s3_url: Optional[str]
    download_progress: int = 0
    user_url: Optional[str]
    filehash_sha256: Optional[str]
    is_public: bool = True
    status: str = "started"
    upload_machine_id: Optional[str]
    upload_type: str
    model_type: str = "checkpoint"
    error_log: Optional[str]
    deleted: bool = False
    is_done: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class DeploymentEnvironment(str, Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    PUBLIC_SHARE = "public-share"
    PRIVATE_SHARE = "private-share"


class WorkflowWithName(BaseModel):
    id: UUID
    name: str
    
class MachineWithName(BaseModel):
    id: UUID
    name: str


class InputModel(BaseModel):
    type: str
    class_type: str
    input_id: str
    default_value: Optional[Union[str, int, float, bool, List[Any]]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    display_name: str = ""
    description: str = ""
    # Add any other fields from value["inputs"] that you want to include

    # You might want to add additional fields based on the specific input types
    # For example:
    enum_options: Optional[List[str]] = Field(
        None, description="Options for enum input type"
    )
    step: Optional[Union[int, float]] = Field(
        None, description="Step for number slider input types"
    )


class OutputModel(BaseModel):
    class_type: str
    output_id: str


class DeploymentModel(BaseModel):
    id: UUID
    user_id: str
    org_id: Optional[str]
    workflow_version_id: UUID
    workflow_id: UUID
    machine_id: UUID
    share_slug: Optional[str]
    description: Optional[str]
    share_options: Optional[Dict[str, Any]]
    showcase_media: Optional[List[Dict[str, Any]]]
    environment: DeploymentEnvironment
    created_at: datetime
    updated_at: datetime
    workflow: Optional[WorkflowWithName] = None
    version: Optional[Any] = None
    machine: Optional[MachineWithName] = None

    input_types: Optional[List[InputModel]] = None
    output_types: Optional[List[OutputModel]] = None

    class Config:
        from_attributes = True


class MachineGPU(str, Enum):
    CPU = "CPU"
    T4 = "T4"
    L4 = "L4"
    A10G = "A10G"
    L40S = "L40S"
    A100 = "A100"
    A100_80GB = "A100-80GB"
    H100 = "H100"


class WorkspaceGPU(str, Enum):
    RTX_4090 = "4090"


class GPUProviderType(str, Enum):
    RUNPOD = "runpod"
    MODAL = "modal"
    COMFY_DEPLOY = "comfy-deploy"


class GPUEventModel(BaseModel):
    id: UUID
    user_id: str
    org_id: Optional[str]
    machine_id: Optional[UUID]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    gpu: Optional[MachineGPU]
    ws_gpu: Optional[WorkspaceGPU]
    gpu_provider: GPUProviderType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    session_timeout: Optional[int] = None
    session_id: Optional[UUID] = None
    modal_function_id: Optional[str] = None
    tunnel_url: Optional[str] = None
    cost_item_title: Optional[str] = None
    cost: Optional[float] = None

    class Config:
        from_attributes = True


class SubscriptionPlanType(str, Enum):
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    CREATOR = "creator"
    BUSINESS = "business"
    WS_BASIC = "ws_basic"
    WS_PRO = "ws_pro"


class SubscriptionPlanStatus(str, Enum):
    ACTIVE = "active"
    DELETED = "deleted"
    PAUSED = "paused"


class SubscriptionStatusType(BaseModel):
    stripe_customer_id: str
    user_id: str
    org_id: Optional[str]
    plan: SubscriptionPlanType
    status: SubscriptionPlanStatus
    subscription_id: str
    subscription_item_plan_id: str
    subscription_item_api_id: str
    cancel_at_period_end: bool
    created_at: datetime
    updated_at: datetime
    trial_end: Optional[datetime]
    trial_start: Optional[datetime]
    last_invoice_timestamp: datetime


class PlanInfo(BaseModel):
    plan: str = "free"
    status: str = "active"
    expires_at: Optional[int] = -1
    spent: Optional[float] = None
    spend_limit: float = Field(default=500.0)
