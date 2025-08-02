import time
import uuid
from deployed.base_app import (
    _ComfyDeployRunner,
    _ComfyDeployRunnerOptimizedModels,
    _ComfyDeployRunnerOptimizedImports,
)
from deployed.comfy_utils import generate_modal_image, optimize_image
import modal

app = modal.App("anycomfyui")

COMFY_DEPENDENCIES = [
    "BennyKok/ComfyUI@2697a11",
    "ty0x2333/ComfyUI-Dev-Utils@0dac07c",
    "BennyKok/comfyui-deploy@a82e315",
]

COMMON_CLS_CONFIG = {
    "allow_concurrent_inputs": 10,
    "container_idle_timeout": 2,
    "image": generate_modal_image(dependencies=COMFY_DEPENDENCIES),
    "gpu": "A100",
    "volumes": {
        "/private_models": modal.Volume.from_name("models_org_2am4LjkQ5IaWGRYMHxGXfHdHcjA")
    },
}

@app.cls(
    **COMMON_CLS_CONFIG,
    enable_memory_snapshot=True,
)
class ComfyDeployRunnerOptimizedModels(_ComfyDeployRunnerOptimizedModels):
    models_to_cache = ["/comfyui/models/checkpoints/v1-5-pruned-emaonly.ckpt"]

@app.cls(
    **COMMON_CLS_CONFIG,
    enable_memory_snapshot=True,
)
class ComfyDeployRunner(_ComfyDeployRunnerOptimizedImports):
    pass

@app.cls(**COMMON_CLS_CONFIG)
class ComfyDeployRunnerNative(_ComfyDeployRunner):
    pass

# @app.cls(**COMMON_CLS_CONFIG)
# class ComfyDeployRunner(_ComfyDeployRunner):
#     native = False

@app.local_entrypoint()
def main():
    workflow_api = """{"3":{"inputs":{"cfg":8,"seed":156680208700286,"model":["4",0],"steps":20,"denoise":1,"negative":["7",0],"positive":["6",0],"scheduler":"normal","latent_image":["5",0],"sampler_name":"euler"},"class_type":"KSampler"},"4":{"inputs":{"ckpt_name":"v1-5-pruned-emaonly.ckpt"},"class_type":"CheckpointLoaderSimple"},"5":{"inputs":{"width":512,"height":512,"batch_size":1},"class_type":"EmptyLatentImage"},"6":{"inputs":{"clip":["4",1],"text":["12",0]},"class_type":"CLIPTextEncode"},"7":{"inputs":{"clip":["4",1],"text":["13",0]},"class_type":"CLIPTextEncode"},"8":{"inputs":{"vae":["4",2],"samples":["3",0]},"class_type":"VAEDecode"},"9":{"inputs":{"images":["8",0],"filename_prefix":"ComfyUI"},"class_type":"SaveImage"},"12":{"inputs":{"input_id":"positive_prompt","default_value":"beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"},"class_type":"ComfyUIDeployExternalText"},"13":{"inputs":{"input_id":"negative_prompt","default_value":"text, watermark"},"class_type":"ComfyUIDeployExternalText"}}"""

    ComfyDeployRunnerOptimizedModels().run.remote(
        {
            "workflow_api_raw": workflow_api,
            "prompt_id": str(uuid.uuid4()),
            "inputs": {
                "positive_prompt": "beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"
            },
        }
    )
