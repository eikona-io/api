from api.utils.inputs import get_inputs_from_workflow_api
from api.utils.outputs import get_outputs_from_workflow
from fastapi import APIRouter
from typing import Any, List, Dict, Literal, Optional, Union

from modal_apps.sd3_5_comfyui import ComfyDeployRunner as SD3_5_ComfyDeployRunner
from modal_apps.flux_dev_comfyui import ComfyDeployRunner as FluxDevComfyDeployRunner
from pydantic import BaseModel

router = APIRouter(tags=["Models"])


class ModelInput(BaseModel):
    input_id: str
    class_type: Union[
        str,
        Literal[
            "ComfyUIDeployExternalText",
            "ComfyUIDeployExternalTextAny",
            "ComfyUIDeployExternalTextSingleLine",
            "ComfyUIDeployExternalImage",
            "ComfyUIDeployExternalImageAlpha",
            "ComfyUIDeployExternalNumber",
            "ComfyUIDeployExternalNumberInt",
            "ComfyUIDeployExternalLora",
            "ComfyUIDeployExternalCheckpoint",
            "ComfyDeployWebscoketImageInput",
            "ComfyUIDeployExternalImageBatch",
            "ComfyUIDeployExternalVideo",
            "ComfyUIDeployExternalBoolean",
            "ComfyUIDeployExternalNumberSlider",
            "ComfyUIDeployExternalNumberSliderInt",
            "ComfyUIDeployExternalEnum",
        ],
    ]
    required: bool
    default_value: Optional[Any] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    display_name: Optional[str] = None
    description: Optional[str] = None


class ModelOutput(BaseModel):
    class_type: Literal["ComfyDeployStdOutputImage", "ComfyDeployStdOutputAny"]
    output_id: str


class Model(BaseModel):
    id: str
    name: str
    is_comfyui: bool = False
    preview_image: Optional[str] = None
    inputs: List[ModelInput]  # Changed from input
    outputs: List[ModelOutput]  # Changed from output


# You might want to move this to a config or separate module
AVAILABLE_MODELS = [
    Model(
        id="flux-dev",
        is_comfyui=True,
        name="Flux Dev",
        preview_image="https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/outputs/runs/b5afa7eb-a15f-4c45-a95c-d5ce89cb537f/image.jpeg",
        inputs=[
            ModelInput(**input, required=True)
            for input in get_inputs_from_workflow_api(
                FluxDevComfyDeployRunner.workflow_api_raw
            )
        ],
        outputs=[
            ModelOutput(**output)
            for output in get_outputs_from_workflow(
                FluxDevComfyDeployRunner.workflow_api_raw
            )
        ],
    ),
    Model(
        id="flux-schnell",
        name="Flux Schnell",
        preview_image="https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/outputs/runs/b5afa7eb-a15f-4c45-a95c-d5ce89cb537f/image.jpeg",
        inputs=[  # Changed from input
            ModelInput(
                input_id="positive_prompt",
                class_type="ComfyUIDeployExternalText",
                required=True,
            ),
        ],
        outputs=[  # Changed from output
            ModelOutput(
                class_type="ComfyDeployStdOutputImage",
                output_id="image",
            )
        ],
    ),
    Model(
        id="sd3-5",
        name="SD3.5",
        is_comfyui=True,
        preview_image="https://comfy-deploy-output.s3.amazonaws.com/outputs/runs/36febfce-3cb6-4220-9447-33003e58d381/ComfyUI_00001_.png",
        inputs=[
            ModelInput(**input, required=True)
            for input in get_inputs_from_workflow_api(
                SD3_5_ComfyDeployRunner.workflow_api_raw
            )
        ],
        outputs=[
            ModelOutput(**output)
            for output in get_outputs_from_workflow(
                SD3_5_ComfyDeployRunner.workflow_api_raw
            )
        ],
    ),
]


@router.get("/models", response_model=List[Model])
async def public_models():
    """Return a list of available public models with their input/output specifications"""
    return AVAILABLE_MODELS
