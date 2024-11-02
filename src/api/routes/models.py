from api.utils.inputs import get_inputs_from_workflow_api
from api.utils.outputs import get_outputs_from_workflow
from fastapi import APIRouter
from typing import Any, List, Dict, Literal, Optional, Union

from comfy_models.workflows import get_all_workflow_configs

# from modal_apps.comfyui.sd3_5_comfyui import (
#     ComfyDeployRunner as SD3_5_ComfyDeployRunner,
# )
# from modal_apps.comfyui.flux_dev_comfyui import (
#     ComfyDeployRunner as FluxDevComfyDeployRunner,
# )
# from modal_apps.comfyui.flux_schnell_comfyui import (
#     ComfyDeployRunner as FluxSchnellComfyDeployRunner,
# )
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


# class WorkflowConfigWithMetaData(WorkflowConfig):
#     is_comfyui: bool = True
#     preview_image: Optional[str] = None


class ModelWithMetadata(Model):
    fal_id: str


# You might want to move this to a config or separate module
AVAILABLE_MODELS = [
    ModelWithMetadata(
        fal_id="fal-ai/flux/dev",
        id="flux-dev",
        is_comfyui=True,
        name="Flux (Dev)",
        preview_image="https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/outputs/runs/b5afa7eb-a15f-4c45-a95c-d5ce89cb537f/image.jpeg",
        inputs=[
            ModelInput(
                input_id="prompt",
                class_type="ComfyUIDeployExternalTextAny",
                required=True,
                description="The prompt to generate an image with",
            )
        ],
        outputs=[],
    ),
]

# AVAILABLE_MODELS = list(get_all_workflow_configs().values())


@router.get("/models", response_model=List[Model])
async def public_models():
    """Return a list of available public models with their input/output specifications"""
    return AVAILABLE_MODELS
