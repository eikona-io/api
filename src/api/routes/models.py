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
    tags: List[str] = []


# class WorkflowConfigWithMetaData(WorkflowConfig):
#     is_comfyui: bool = True
#     preview_image: Optional[str] = None


class ModelWithMetadata(Model):
    fal_id: Optional[str] = None
    cost_per_megapixel: Optional[float] = None


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
seed = ModelInput(
    input_id="seed",
    class_type="ComfyUIDeployExternalSeed",
    min_value=0,
    max_value=2147483647,
    required=True,
    default_value=None,
)
num_inference_steps = ModelInput(
    display_name="Num Inference Steps",
    input_id="num_inference_steps",
    class_type="ComfyUIDeployExternalNumberSliderInt",
    required=False,
    default_value=28,
    min_value=1,
    max_value=50,
)
guidance_scale = ModelInput(
    display_name="Guidance scale (CFG)",
    input_id="guidance_scale",
    class_type="ComfyUIDeployExternalNumberSlider",
    required=False,
    default_value=3.5,
    min_value=1,
    max_value=20,
)
colors = ModelInput(
    input_id="colors",
    class_type="ComfyUIDeployExternalColor",
    required=False,
    default_value=None,
)
flux_prompt = ModelInput(
    input_id="prompt",
    class_type="ComfyUIDeployExternalText",
    required=True,
    default_value='Extreme close-up of a single tiger eye, direct frontal view. Detailed iris and pupil. Sharp focus on eye texture and color. Natural lighting to capture authentic eye shine and depth. The word "FLUX" is painted over it in big, white brush strokes with visible texture.',
    description="The prompt to generate an image with",
)
raw = ModelInput(
    input_id="raw",
    class_type="ComfyUIDeployExternalBoolean",
    required=True,
    default_value=False,
)
duration = ModelInput(
    input_id="duration",
    class_type="ComfyUIDeployExternalEnum",
    required=True,
    default_value="4",
    enum_values=[
        "4",
        "6",
    ],
)
prompt_enhancer = ModelInput(
    input_id="prompt_enhancer",
    class_type="ComfyUIDeployExternalBoolean",
    required=True,
    default_value=True,
)

prompt = ModelInput(
    input_id="prompt",
    class_type="ComfyUIDeployExternalText",
    required=True,
    default_value='Extreme close-up of a single tiger eye, direct frontal view. Detailed iris and pupil. Sharp focus on eye texture and color. Natural lighting to capture authentic eye shine and depth. The word "SD3.5" is painted over it in big, white brush strokes with visible texture.',
    description="The prompt to generate an image with",
)
recraft_style = ModelInput(
    input_id="recraft_style",
    class_type="ComfyUIDeployExternalEnum",
    required=True,
    default_value="realistic_image",
    enum_values=[
        "any",
        "realistic_image",
        "digital_illustration",
        # "vector_illustration",
        "realistic_image/b_and_w",
        "realistic_image/hard_flash",
        "realistic_image/hdr",
        "realistic_image/natural_light",
        "realistic_image/studio_portrait",
        "realistic_image/enterprise",
        "realistic_image/motion_blur",
        "digital_illustration/pixel_art",
    ],
)

aspect_ratio = ModelInput(
    input_id="aspect_ratio",
    class_type="ComfyUIDeployExternalEnum",
    required=True,
    default_value="16:9",
    enum_values=[
        "16:9",
        "9:16",
        "4:3",
        "3:4",
        "21:9",
        "9:21",
    ],
)
image_url = ModelInput(
    input_id="image_url",
    class_type="ComfyUIDeployExternalImage",
    required=True,
    default_value=None,
)

input_image_urls = ModelInput(
    input_id="input_image_urls",
    class_type="ComfyUIDeployExternalImageBatch",
    required=True,
    default_value=None,
)

# You might want to move this to a config or separate module
AVAILABLE_MODELS = [
    ModelWithMetadata(
        fal_id="fal-ai/flux/dev",
        id="flux-dev",
        name="Flux (Dev)",
        preview_image="https://fal.media/files/koala/LmLyc8U4EVekGyGFWan1M.png",
        inputs=[flux_prompt, image_size, seed, num_inference_steps, guidance_scale],
        outputs=[],
        tags=["flux", "image"],
        cost_per_megapixel=0.025,
    ),
    ModelWithMetadata(
        fal_id="fal-ai/flux/schnell",
        id="flux-schnell",
        name="Flux (Schnell)",
        preview_image="https://fal.media/files/panda/UtTYMhOHimr0rEYq20dFP.png",
        inputs=[flux_prompt, image_size],
        outputs=[],
        tags=["flux", "image"],
        cost_per_megapixel=0.003
    ),
    ModelWithMetadata(
        fal_id="fal-ai/omnigen-v1",
        id="omnigen-v1",
        name="Omnigen V1",
        preview_image="https://fal.media/files/elephant/YINEd790XmHDu-YGsG3gJ.jpeg",
        inputs=[prompt, input_image_urls, image_size, seed, guidance_scale, num_inference_steps],
        outputs=[],
        tags=["omnigen", "image"],
        cost_per_megapixel=0.1,
    ),
    ModelWithMetadata(
        fal_id="fal-ai/flux-pro/v1.1-ultra",
        id="flux-pro-v1.1-ultra",
        name="Flux V1.1 (Pro) Ultra",
        preview_image="https://fal.media/files/kangaroo/qur7RE3oRed27VmCcSZB6_03c743bc8ab544f28978eb700df1afab.jpg",
        inputs=[flux_prompt, seed, aspect_ratio, raw],
        outputs=[],
        tags=["flux", "image"],
        cost_per_megapixel=0.06,
    ),
    ModelWithMetadata(
        fal_id="fal-ai/stable-diffusion-v35-medium",
        id="sd-v35-medium",
        name="Stable Diffusion V3.5 (Medium)",
        preview_image="https://comfy-deploy-output.s3.amazonaws.com/outputs/runs/36febfce-3cb6-4220-9447-33003e58d381/ComfyUI_00001_.png",
        inputs=[prompt, image_size],
        outputs=[],
        tags=["sd3.5", "image"],
        cost_per_megapixel=0.02
    ),
    ModelWithMetadata(
        fal_id="fal-ai/stable-diffusion-v35-large",
        id="sd-v35-large",
        name="Stable Diffusion V3.5 (Large)",
        preview_image="https://fal.media/files/zebra/yr8dajXZ9LaIyTxpVlb3n.jpeg",
        inputs=[prompt, image_size],
        outputs=[],
        tags=["sd3.5", "image"],
        cost_per_megapixel=0.065
    ),
    ModelWithMetadata(
        fal_id="fal-ai/recraft-v3",
        id="recraft-v3",
        name="Recraft V3",
        preview_image="https://fal.media/files/penguin/-qx-N4DHuAP9RA_CWAfSt_image.webp",
        inputs=[prompt, image_size, recraft_style, colors],
        outputs=[],
        tags=["recraft", "image"],
        cost_per_megapixel=0.04,
    ),
    # ModelWithMetadata(
    #     fal_id="fal-ai/luma-dream-machine",
    #     id="luma-dream-machine",
    #     name="Luma Dream Machine",
    #     preview_image="https://v2.fal.media/files/807e842c734f4127a36de9262a2d292c_output.mp4",
    #     inputs=[prompt, aspect_ratio],
    #     outputs=[],
    #     tags=["luma", "video"],
    #     cost_per_megapixel=0.5
    # ),
    # ModelWithMetadata(
    #     fal_id="fal-ai/haiper-video-v2",
    #     id="haiper-video-v2",
    #     name="Haiper 2.0 Video",
    #     preview_image="https://fal.media/files/koala/ki_nVspVCkT8JpgrjKdqC_output.mp4",
    #     inputs=[prompt, duration, prompt_enhancer, seed],
    #     outputs=[],
    #     tags=["haiper", "video"],
    #     cost_per_megapixel=0.04,
    # ),
    ModelWithMetadata(
        fal_id="fal-ai/flux/dev/image-to-image",
        id="flux-dev-image-to-image",
        name="Flux (Dev) Image to Image",
        preview_image="https://fal.media/files/rabbit/mYYiweNeGvoYb1jZX3erl.png",
        inputs=[flux_prompt, image_url, image_size, seed, num_inference_steps, guidance_scale],
        outputs=[],
        tags=["flux", "image"],
        cost_per_megapixel=0.03,
    ),
    # ModelWithMetadata(
    #     fal_id="fal-ai/aura-flow",
    #     id="aura-flow",
    #     name="AuraFlow",
    #     preview_image="https://fal.media/files/kangaroo/HPWc0iotMy0wLxp7SbGtE_1a7fdbd386a94da1960bf2c41af14050.png",
    #     inputs=[prompt, seed, guidance_scale, num_inference_steps],
    #     outputs=[],
    #     tags=["aura", "image"],
    #     cost_per_megapixel=0.02,
    # ),
    ModelWithMetadata(
        fal_id="fal-ai/minimax-video/image-to-video",
        id="minimax-video-image-to-video",
        name="MiniMax (Hailuo AI) Video [Image to Video]",
        preview_image="https://fal.media/files/penguin/eD9AqUrqCzFkQXSd_xXG2_output.mp4",
        inputs=[prompt, image_url],
        outputs=[],
        tags=["minimax", "video"],
        cost_per_megapixel=0.5,
    ),
]

# AVAILABLE_MODELS = list(get_all_workflow_configs().values())


@router.get("/models", response_model=List[ModelWithMetadata])
async def public_models():
    """Return a list of available public models with their input/output specifications"""
    return AVAILABLE_MODELS
