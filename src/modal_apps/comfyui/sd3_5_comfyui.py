import time
import uuid
from deployed.base_app import (
    _ComfyDeployRunner,
    _ComfyDeployRunnerModelsDownloadOptimzedImports,
    _ComfyDeployRunnerOptimizedModels,
    _ComfyDeployRunnerOptimizedImports,
)
from deployed.comfy_utils import generate_modal_image, optimize_image
import modal
from modal_apps.comfyui.shared import SHARED_CLS_CONFIG, get_volumes

app = modal.App("sd3-5")

COMFY_DEPENDENCIES = [
    "BennyKok/ComfyUI@2697a11",
    "ty0x2333/ComfyUI-Dev-Utils@0dac07c",
    "BennyKok/comfyui-deploy@a82e315",
]

SAFE_MODEL_NAME = "sd3_5".replace("/", "-")  # Replace slashes with dashes

COMMON_CLS_CONFIG = {
    "image": generate_modal_image(dependencies=COMFY_DEPENDENCIES),
    "volumes": get_volumes(SAFE_MODEL_NAME),
    **SHARED_CLS_CONFIG,
}


@app.cls(
    **COMMON_CLS_CONFIG,
    enable_memory_snapshot=True,
    experimental_options={"gpu_memory_snapshot": True},
)
class ComfyDeployRunner(_ComfyDeployRunnerModelsDownloadOptimzedImports):
    # skip_workflow_api_validation = True

    # run_twice = True

    workflow_api_raw = """{"3":{"inputs":{"cfg":5.45,"seed":844390075629552,"model":["57",0],"steps":20,"denoise":1,"negative":["40",0],"positive":["16",0],"scheduler":"sgm_uniform","latent_image":["53",0],"sampler_name":"euler"},"class_type":"KSampler"},"4":{"inputs":{"ckpt_name":"sd3.5_large.safetensors"},"class_type":"CheckpointLoaderSimple"},"8":{"inputs":{"vae":["4",2],"samples":["3",0]},"class_type":"VAEDecode"},"9":{"inputs":{"images":["8",0],"filename_prefix":"ComfyUI"},"class_type":"SaveImage"},"16":{"inputs":{"clip":["43",0],"text":["56",0]},"class_type":"CLIPTextEncode"},"40":{"inputs":{"clip":["43",0],"text":""},"class_type":"CLIPTextEncode"},"41":{"inputs":{"type":"sd3","clip_name":"t5xxl_fp16.safetensors"},"class_type":"CLIPLoader"},"42":{"inputs":{"type":"sd3","clip_name1":"clip_l.safetensors","clip_name2":"clip_g.safetensors"},"class_type":"DualCLIPLoader"},"43":{"inputs":{"clip_name1":"clip_l.safetensors","clip_name2":"clip_g.safetensors","clip_name3":"t5xxl_fp16.safetensors"},"class_type":"TripleCLIPLoader"},"53":{"inputs":{"width":1024,"height":1024,"batch_size":1},"class_type":"EmptySD3LatentImage"},"54":{"inputs":{"ckpt_name":"sd3.5_large.safetensors"},"class_type":"CheckpointLoaderSimple"},"56":{"inputs":{"input_id":"positive_prompt","description":"","display_name":"","default_value":"a bottle with a rainbow galaxy inside it on top of a wooden table on a snowy mountain top with the ocean and clouds in the background with a shot glass beside containing darkness beside a snow sculpture in the shape of a fox"},"class_type":"ComfyUIDeployExternalText"},"57":{"inputs":{"model":["4",0],"backend":"inductor"},"class_type":"TorchCompileModel"}}"""
    # workflow_api_raw = """{"3":{"inputs":{"cfg":5.45,"seed":844390075629552,"model":["4",0],"steps":20,"denoise":1,"negative":["40",0],"positive":["16",0],"scheduler":"sgm_uniform","latent_image":["53",0],"sampler_name":"euler"},"class_type":"KSampler"},"4":{"inputs":{"ckpt_name":"sd3.5_large.safetensors"},"class_type":"CheckpointLoaderSimple"},"8":{"inputs":{"vae":["4",2],"samples":["3",0]},"class_type":"VAEDecode"},"9":{"inputs":{"images":["8",0],"filename_prefix":"ComfyUI"},"class_type":"SaveImage"},"16":{"inputs":{"clip":["43",0],"text":["56",0]},"class_type":"CLIPTextEncode"},"40":{"inputs":{"clip":["43",0],"text":""},"class_type":"CLIPTextEncode"},"41":{"inputs":{"type":"sd3","clip_name":"t5xxl_fp16.safetensors"},"class_type":"CLIPLoader"},"42":{"inputs":{"type":"sd3","clip_name1":"clip_l.safetensors","clip_name2":"clip_g.safetensors"},"class_type":"DualCLIPLoader"},"43":{"inputs":{"clip_name1":"clip_l.safetensors","clip_name2":"clip_g.safetensors","clip_name3":"t5xxl_fp16.safetensors"},"class_type":"TripleCLIPLoader"},"53":{"inputs":{"width":1024,"height":1024,"batch_size":1},"class_type":"EmptySD3LatentImage"},"54":{"inputs":{"ckpt_name":"sd3.5_large.safetensors"},"class_type":"CheckpointLoaderSimple"},"56":{"inputs":{"input_id":"positive_prompt","description":"","display_name":"","default_value":"a bottle with a rainbow galaxy inside it on top of a wooden table on a snowy mountain top with the ocean and clouds in the background with a shot glass beside containing darkness beside a snow sculpture in the shape of a fox"},"class_type":"ComfyUIDeployExternalText"}}"""

    models_to_cache = [
        "/comfyui/models/checkpoints/sd3.5_large.safetensors",
        "/comfyui/models/clip/clip_l.safetensors",
        "/comfyui/models/clip/clip_g.safetensors",
        "/comfyui/models/clip/t5xxl_fp16.safetensors",
    ]


@app.local_entrypoint()
def main():
    ComfyDeployRunner().run.remote(
        {
            "prompt_id": str(uuid.uuid4()),
            "inputs": {
                "positive_prompt": "beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"
            },
        }
    )
