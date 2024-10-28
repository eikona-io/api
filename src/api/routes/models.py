from fastapi import APIRouter
from typing import Any, List, Dict, Literal, Optional

from pydantic import BaseModel

router = APIRouter(tags=["Models"])


class ModelInput(BaseModel):
    input_id: str
    class_type: Literal[
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
    inputs: List[ModelInput]    # Changed from input
    outputs: List[ModelOutput]  # Changed from output


# You might want to move this to a config or separate module
AVAILABLE_MODELS = [
    Model(
        id="flux-dev",
        name="Flux Dev",
        inputs=[                # Changed from input
            ModelInput(
                input_id="positive_prompt",
                class_type="ComfyUIDeployExternalText",
                required=True,
            ),
        ],
        outputs=[              # Changed from output
            ModelOutput(
                class_type="ComfyDeployStdOutputImage",
                output_id="image",
            )
        ],
    ),
    Model(
        id="flux-schnell",
        name="Flux Schnell",
        inputs=[                # Changed from input
            ModelInput(
                input_id="positive_prompt",
                class_type="ComfyUIDeployExternalText",
                required=True,
            ),
        ],
        outputs=[              # Changed from output
            ModelOutput(
                class_type="ComfyDeployStdOutputImage",
                output_id="image",
            )
        ],
    ),
]

@router.get("/models", response_model=List[Model])
async def public_models():
    """Return a list of available public models with their input/output specifications"""
    return AVAILABLE_MODELS
