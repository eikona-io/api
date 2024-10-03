import json
import logging
from fastapi import APIRouter, Depends, Request
from typing import Any, Dict, List, Optional, Union
from .types import DeploymentModel, DeploymentEnvironment
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import select
from api.models import Deployment, Workflow
from api.database import get_db
from fastapi.responses import JSONResponse
from sqlalchemy.orm import joinedload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Deployments"])


@router.get(
    "/deployments",
    response_model=List[DeploymentModel],
    openapi_extra={
        "x-speakeasy-name-override": "list",
    },
)
async def get_deployments(
    request: Request,
    environment: Optional[DeploymentEnvironment] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Deployment).options(
        joinedload(Deployment.workflow).load_only(Workflow.name),
        joinedload(Deployment.version),
    )

    if environment is not None:
        query = query.where(Deployment.environment == environment)

    query = query.apply_org_check(request)

    result = await db.execute(query)
    deployments = result.scalars().all()

    deployments_data = []
    for deployment in deployments:
        deployment_dict = deployment.to_dict()
        workflow_api = deployment.version.workflow_api if deployment.version else None
        inputs = get_inputs_from_workflow_api(workflow_api)
        logger.info(inputs)
        if inputs:
            deployment_dict["input_types"] = inputs
        deployments_data.append(deployment_dict)

    return deployments_data


custom_input_nodes: Dict[str, Dict[str, str]] = {
    "ComfyUIDeployExternalText": {
        "type": "string",
        "description": "Multi-line text input",
    },
    "ComfyUIDeployExternalTextAny": {"type": "string", "description": "Any text input"},
    "ComfyUIDeployExternalTextSingleLine": {
        "type": "string",
        "description": "Single-line text input",
    },
    "ComfyUIDeployExternalImage": {"type": "string", "description": "Public image URL"},
    "ComfyUIDeployExternalImageAlpha": {
        "type": "string",
        "description": "Public image URL with alpha channel",
    },
    "ComfyUIDeployExternalNumber": {
        "type": "float",
        "description": "Floating-point number input",
    },
    "ComfyUIDeployExternalNumberInt": {
        "type": "integer",
        "description": "Integer number input",
    },
    "ComfyUIDeployExternalLora": {
        "type": "string",
        "description": "Public LoRA download URL",
    },
    "ComfyUIDeployExternalCheckpoint": {
        "type": "string",
        "description": "Public checkpoint download URL",
    },
    "ComfyDeployWebscoketImageInput": {
        "type": "binary",
        "description": "Websocket image input",
    },
    "ComfyUIDeployExternalImageBatch": {
        "type": "string",
        "description": "Array of image URLs",
    },
    "ComfyUIDeployExternalVideo": {"type": "string", "description": "Public video URL"},
    "ComfyUIDeployExternalBoolean": {"type": "boolean", "description": "Boolean input"},
    "ComfyUIDeployExternalNumberSlider": {
        "type": "float",
        "description": "Floating-point number slider",
    },
    "ComfyUIDeployExternalNumberSliderInt": {
        "type": "integer",
        "description": "Integer number slider",
    },
    "ComfyUIDeployExternalEnum": {
        "type": "string",
        "description": "Enumerated string options",
    },
}

# This is a type hint for the CustomInputNodesTypeMap
CustomInputNodesTypeMap = Dict[str, Union[str, int, float, bool, List[str], bytes]]

# Define InputsType as a string literal type (closest Python equivalent)
InputsType = str

# Create the list of input types
input_types_list: List[InputsType] = list(custom_input_nodes.keys())


def get_inputs_from_workflow_api(
    workflow_api: Optional[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    if not workflow_api:
        return None

    inputs = []
    for _, value in workflow_api.items():
        if not value.get("class_type"):
            continue

        node_type = custom_input_nodes.get(value["class_type"])

        if node_type:
            input_id = value["inputs"].get("input_id", "")
            default_value = value["inputs"].get("default_value")

            input_data = {
                **value["inputs"],
                "class_type": value["class_type"],
                "type": node_type.get("type"),
                "input_id": input_id,
                "default_value": default_value,
                "min_value": value["inputs"].get("min_value"),
                "max_value": value["inputs"].get("max_value"),
                "display_name": value["inputs"].get("display_name", ""),
                "description": value["inputs"].get("description", ""),
            }
            inputs.append(input_data)

    return inputs if inputs else None
