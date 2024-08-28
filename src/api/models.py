import inspect
from sqlalchemy import (
    Column,
    String,
    Enum,
    DateTime,
    Boolean,
    MetaData,
    Float,
    JSON,
    ForeignKey,
    Integer,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.inspection import inspect as sqlalchemy_inspect
import uuid
from datetime import datetime
import json
from datetime import timezone

Base = declarative_base()

metadata = MetaData(schema="comfyui_deploy")


class SerializableMixin:
    def to_dict(self):
        try:
            return {
                c.key: self._serialize_value(getattr(self, c.key))
                for c in sqlalchemy_inspect(self.__class__).mapper.column_attrs
                if hasattr(self, c.key)
            }
        except Exception as e:
            print(f"Error in to_dict: {str(e)}")
            return {}

    def _serialize_value(self, value):
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, datetime):
            # .replace(tzinfo=timezone.utc)
            return value.isoformat()[:-3]+'Z'
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        return value

    def to_json(self):
        return json.dumps(self.to_dict())


class Workflow(SerializableMixin, Base):
    __tablename__ = "workflows"
    metadata = metadata

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="cascade"), nullable=False)
    org_id = Column(String)
    name = Column(String, nullable=False)
    selected_machine_id = Column(
        UUID(as_uuid=True), ForeignKey("machines.id", ondelete="set null")
    )
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    pinned = Column(Boolean, nullable=False, default=False)

    # user = relationship("User", back_populates="workflows")
    # versions = relationship("WorkflowVersion", back_populates="workflow_rel")
    # deployments = relationship("Deployment", back_populates="workflow")
    # selected_machine = relationship("Machine", foreign_keys=[selected_machine_id])
    # machine = relationship("Machine", back_populates="workflows", foreign_keys=[selected_machine_id])
    # runs = relationship("WorkflowRun", back_populates="workflow")


class WorkflowVersion(SerializableMixin, Base):
    __tablename__ = "workflow_versions"
    metadata = metadata

    workflow_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflows.id", ondelete="cascade"),
        nullable=False,
    )
    id = Column(UUID(as_uuid=True), primary_key=True)
    workflow = Column(JSON)
    workflow_api = Column(JSON)
    user_id = Column(String, ForeignKey("users.id", ondelete="set null"))
    comment = Column(String)
    version = Column(Integer, nullable=False)
    snapshot = Column(JSON)
    dependencies = Column(JSON)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    # workflow_rel = relationship("Workflow", back_populates="versions")
    # user = relationship("User", back_populates="workflow_versions")
    
    
def lazy_utc_now():
    return datetime.now(tz=timezone.utc)


class WorkflowRun(SerializableMixin, Base):
    __tablename__ = "workflow_runs"
    metadata = metadata

    id = Column(UUID(as_uuid=True), primary_key=True)
    workflow_version_id = Column(UUID(as_uuid=True), nullable=True)
    workflow_inputs = Column(JSON)
    workflow_id = Column(UUID(as_uuid=True))
    workflow_api = Column(JSON)
    machine_id = Column(UUID(as_uuid=True), nullable=True)
    origin = Column(
        Enum(
            "manual",
            "api",
            "public-share",
            "public-template",
            "workspace",
            "workspace-v2",
            name="workflow_run_origin",
        ),
        nullable=False,
        default="api",
    )
    status = Column(
        Enum(
            "not-started",
            "running",
            "uploading",
            "success",
            "failed",
            "started",
            "queued",
            "timeout",
            "cancelled",
            name="workflow_run_status",
        ),
        nullable=False,
        default="not-started",
    )
    ended_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, default=lazy_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lazy_utc_now)
    queued_at = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    gpu_event_id = Column(String)
    gpu = Column(Enum("T4", "L4", "A10G", "A100", "A100-80GB", "H100", name="machine_gpu"))
    machine_version = Column(String)
    machine_type = Column(
        Enum(
            "classic",
            "runpod-serverless",
            "modal-serverless",
            "comfy-deploy-serverless",
            "workspace",
            "workspace-v2",
            name="machine_type",
        )
    )
    modal_function_call_id = Column(String)
    user_id = Column(String)
    org_id = Column(String)
    run_log = Column(JSON)
    live_status = Column(String)
    progress = Column(Float, nullable=False, default=0)
    is_realtime = Column(Boolean, nullable=False, default=False)
    webhook = Column(String)
    webhook_status = Column(
        Enum("success", "failed", "not-started", "running", name="webhook_status")
    )
    webhook_intermediate_status = Column(Boolean, nullable=False, default=False)
    
    batch_id = Column(UUID(as_uuid=True))

    # workflow = relationship("Workflow", back_populates="runs")
    outputs = relationship("WorkflowRunOutput", back_populates="run")


class WorkflowRunOutput(SerializableMixin, Base):
    __tablename__ = "workflow_run_outputs"
    metadata = metadata

    id = Column(UUID(as_uuid=True), primary_key=True)
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflow_runs.id", ondelete="cascade"),
        nullable=False,
    )
    data = Column(JSON)
    node_meta = Column(JSON)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    run = relationship("WorkflowRun", back_populates="outputs")


class APIKey(SerializableMixin, Base):
    __tablename__ = "api_keys"
    metadata = metadata

    id = Column(UUID(as_uuid=True), primary_key=True)
    key = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    org_id = Column(String)
    revoked = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    # user = relationship("User", back_populates="api_keys")

    def is_revoked(self):
        return self.revoked


class User(SerializableMixin, Base):
    __tablename__ = "users"
    metadata = metadata

    id = Column(String, primary_key=True)
    # Add other user fields as needed

    # api_keys = relationship("APIKey", back_populates="user")
    # workflows = relationship("Workflow", back_populates="user")
    # workflow_versions = relationship("WorkflowVersion", back_populates="user")


class Deployment(SerializableMixin, Base):
    __tablename__ = "deployments"
    metadata = metadata

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="cascade"), nullable=False)
    org_id = Column(String)
    workflow_version_id = Column(
        UUID(as_uuid=True), ForeignKey("workflow_versions.id"), nullable=False
    )
    workflow_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflows.id", ondelete="cascade"),
        nullable=False,
    )
    machine_id = Column(UUID(as_uuid=True), ForeignKey("machines.id"), nullable=False)
    share_slug = Column(String, unique=True)
    description = Column(String)
    share_options = Column(JSON)
    showcase_media = Column(JSON)
    environment = Column(
        Enum(
            "staging",
            "production",
            "public-share",
            "private-share",
            name="deployment_environment",
        ),
        nullable=False,
    )
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    # machine = relationship("Machine")
    # version = relationship("WorkflowVersion")
    # workflow = relationship("Workflow")
    # user = relationship("User")


class Machine(SerializableMixin, Base):
    __tablename__ = "machines"
    metadata = metadata

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="cascade"), nullable=False)
    name = Column(String, nullable=False)
    org_id = Column(String)
    endpoint = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    disabled = Column(Boolean, nullable=False, default=False)
    auth_token = Column(String)
    type = Column(
        Enum(
            "classic",
            "runpod-serverless",
            "modal-serverless",
            "comfy-deploy-serverless",
            "workspace",
            "workspace-v2",
            name="machine_type",
        ),
        nullable=False,
        default="classic",
    )
    status = Column(
        Enum(
            "not-started",
            "ready",
            "building",
            "error",
            "running",
            "paused",
            "starting",
            name="machine_status",
        ),
        nullable=False,
        default="ready",
    )
    static_assets_status = Column(
        Enum(
            "not-started",
            "ready",
            "building",
            "error",
            "running",
            "paused",
            "starting",
            name="machine_status",
        ),
        nullable=False,
        default="not-started",
    )
    machine_version = Column(String)
    machine_builder_version = Column(
        Enum("2", "3", name="machine_builder_version"), default="2"
    )
    snapshot = Column(JSON)
    models = Column(JSON)
    gpu = Column(Enum("T4", "L4", "A10G", "A100", "A100-80GB", "H100", name="machine_gpu"))
    ws_gpu = Column(Enum("4090", name="workspace_machine_gpu"))
    pod_id = Column(String)
    base_docker_image = Column(String)
    allow_concurrent_inputs = Column(Integer, default=1)
    concurrency_limit = Column(Integer, default=2)
    legacy_mode = Column(Boolean, nullable=False, default=False)
    ws_timeout = Column(Integer, default=2)
    run_timeout = Column(Integer, nullable=False, default=300)
    idle_timeout = Column(Integer, nullable=False, default=60)
    build_machine_instance_id = Column(String)
    build_log = Column(String)
    modal_app_id = Column(String)
    target_workflow_id = Column(
        UUID(as_uuid=True), ForeignKey("workflows.id", ondelete="set null")
    )
    dependencies = Column(JSON)
    extra_docker_commands = Column(JSON)
    install_custom_node_with_gpu = Column(Boolean, default=False)
    deleted = Column(Boolean, nullable=False, default=False)
    keep_warm = Column(Integer, nullable=False, default=0)
    allow_background_volume_commits = Column(Boolean, nullable=False, default=False)
    gpu_workspace = Column(Boolean, nullable=False, default=False)
    docker_command_steps = Column(JSON)
    comfyui_version = Column(String)
    python_version = Column(String)
    extra_args = Column(String)
    prestart_command = Column(String)
    retrieve_static_assets = Column(Boolean, default=False)
    object_info = Column(JSON)
    object_info_str = Column(String)
    filename_list_cache = Column(JSON)
    extensions = Column(JSON)

    # target_workflow = relationship("Workflow", foreign_keys=[target_workflow_id])
    # workflows = relationship("Workflow", back_populates="machine", foreign_keys=[Workflow.selected_machine_id])