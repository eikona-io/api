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
    enum_values: Optional[List[str]] = None


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


image_size = ModelInput(
    input_id="image_size",
    class_type="ComfyUIDeployExternalEnum",
    required=True,
    default_value="square_hd",
    enum_values=[
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
    ],
)
flux_prompt = ModelInput(
    input_id="prompt",
    class_type="ComfyUIDeployExternalText",
    required=True,
    default_value='Extreme close-up of a single tiger eye, direct frontal view. Detailed iris and pupil. Sharp focus on eye texture and color. Natural lighting to capture authentic eye shine and depth. The word "FLUX" is painted over it in big, white brush strokes with visible texture.',
    description="The prompt to generate an image with",
)

prompt = ModelInput(
    input_id="prompt",
    class_type="ComfyUIDeployExternalText",
    required=True,
    default_value='Extreme close-up of a single tiger eye, direct frontal view. Detailed iris and pupil. Sharp focus on eye texture and color. Natural lighting to capture authentic eye shine and depth. The word "SD3.5" is painted over it in big, white brush strokes with visible texture.',
    description="The prompt to generate an image with",
)


# You might want to move this to a config or separate module
AVAILABLE_MODELS = [
    ModelWithMetadata(
        fal_id="fal-ai/flux/schnell",
        id="flux-schnell",
        name="Flux (Schnell)",
        preview_image="https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/outputs/runs/b5afa7eb-a15f-4c45-a95c-d5ce89cb537f/image.jpeg",
        inputs=[flux_prompt, image_size],
        outputs=[],
    ),
    ModelWithMetadata(
        fal_id="fal-ai/flux/dev",
        id="flux-dev",
        name="Flux (Dev)",
        preview_image="https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/outputs/runs/b5afa7eb-a15f-4c45-a95c-d5ce89cb537f/image.jpeg",
        inputs=[flux_prompt, image_size],
        outputs=[],
    ),
    ModelWithMetadata(
        fal_id="fal-ai/stable-diffusion-v35-medium",
        id="sd-v35-medium",
        name="SD V3.5 (Medium)",
        preview_image="https://comfy-deploy-output.s3.amazonaws.com/outputs/runs/36febfce-3cb6-4220-9447-33003e58d381/ComfyUI_00001_.png",
        inputs=[prompt, image_size],
        outputs=[],
    ),
    ModelWithMetadata(
        fal_id="fal-ai/stable-diffusion-v35-large",
        id="sd-v35-large",
        name="SD V3.5 (Large)",
        preview_image="https://comfy-deploy-output.s3.amazonaws.com/outputs/runs/36febfce-3cb6-4220-9447-33003e58d381/ComfyUI_00001_.png",
        inputs=[prompt, image_size],
        outputs=[],
    ),
]

# AVAILABLE_MODELS = list(get_all_workflow_configs().values())


@router.get("/models", response_model=List[Model])
async def public_models():
    """Return a list of available public models with their input/output specifications"""
    return AVAILABLE_MODELS
