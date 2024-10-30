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

app = modal.App("flux-schnell")

COMFY_DEPENDENCIES = [
    "BennyKok/ComfyUI@2697a11",
    "ty0x2333/ComfyUI-Dev-Utils@0dac07c",
    "BennyKok/comfyui-deploy@a82e315",
]

SAFE_MODEL_NAME = "flux_schnell".replace("/", "-")  # Replace slashes with dashes
volumes = {  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
    "/root/.nv": modal.Volume.from_name(
        "nv-cache-" + SAFE_MODEL_NAME, create_if_missing=True
    ),
    "/root/.triton": modal.Volume.from_name(
        "triton-cache-" + SAFE_MODEL_NAME, create_if_missing=True
    ),
    "/root/.inductor-cache": modal.Volume.from_name(
        "inductor-cache-" + SAFE_MODEL_NAME, create_if_missing=True
    ),
    # "/models-cache": modal.Volume.lookup("models_org_2am4LjkQ5IaWGRYMHxGXfHdHcjA"),
    "/private_models": modal.Volume.lookup("models_org_2am4LjkQ5IaWGRYMHxGXfHdHcjA"),
}

COMMON_CLS_CONFIG = {
    "allow_concurrent_inputs": 10,
    "container_idle_timeout": 300,
    "timeout": 60 * 60,  # leave plenty of time for compilation
    "image": generate_modal_image(dependencies=COMFY_DEPENDENCIES),
    "gpu": "H100",
    "secrets": [modal.Secret.from_name("hf-models-download")],
    # "volumes": {
    #     "/private_models": modal.Volume.lookup("models_org_2am4LjkQ5IaWGRYMHxGXfHdHcjA")
    # },
    "volumes": volumes,
}


@app.cls(
    **COMMON_CLS_CONFIG,
    enable_memory_snapshot=True,
)
class ComfyDeployRunner(_ComfyDeployRunnerModelsDownloadOptimzedImports):
    # @modal.build()
    # def build(self):
    #     self.download_models()

    # skip_workflow_api_validation = True
    
    # run_twice = True

    workflow_api_raw = """{"6":{"inputs":{"clip":["11",0],"text":["38",0]},"class_type":"CLIPTextEncode"},"8":{"inputs":{"vae":["10",0],"samples":["13",0]},"class_type":"VAEDecode"},"9":{"inputs":{"images":["8",0],"filename_prefix":"ComfyUI"},"class_type":"SaveImage"},"10":{"inputs":{"vae_name":"ae.sft"},"class_type":"VAELoader"},"11":{"inputs":{"type":"flux","clip_name1":"t5xxl_fp16.safetensors","clip_name2":"clip_l.safetensors"},"class_type":"DualCLIPLoader"},"12":{"inputs":{"unet_name":"flux1-schnell.sft","weight_dtype":"fp8_e4m3fn_fast"},"class_type":"UNETLoader"},"13":{"inputs":{"noise":["25",0],"guider":["22",0],"sigmas":["17",0],"sampler":["16",0],"latent_image":["27",0]},"class_type":"SamplerCustomAdvanced"},"16":{"inputs":{"sampler_name":"euler"},"class_type":"KSamplerSelect"},"17":{"inputs":{"model":["30",0],"steps":2,"denoise":1,"scheduler":"simple"},"class_type":"BasicScheduler"},"22":{"inputs":{"model":["30",0],"conditioning":["26",0]},"class_type":"BasicGuider"},"25":{"inputs":{"noise_seed":219670278747233},"class_type":"RandomNoise"},"26":{"inputs":{"guidance":3.5,"conditioning":["6",0]},"class_type":"FluxGuidance"},"27":{"inputs":{"width":1024,"height":1024,"batch_size":1},"class_type":"EmptySD3LatentImage"},"30":{"inputs":{"model":["39",0],"width":1024,"height":1024,"max_shift":1.15,"base_shift":0.5},"class_type":"ModelSamplingFlux"},"38":{"inputs":{"input_id":"positive_prompt","description":"","display_name":"","default_value":"cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open holding a fancy black forest cake with candles on top in the kitchen of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere"},"class_type":"ComfyUIDeployExternalText"},"39":{"inputs":{"model":["12",0],"backend":"inductor"},"class_type":"TorchCompileModel"}}"""
    # workflow_api_raw = """{"6":{"inputs":{"clip":["11",0],"text":["38",0]},"class_type":"CLIPTextEncode"},"8":{"inputs":{"vae":["10",0],"samples":["13",0]},"class_type":"VAEDecode"},"9":{"inputs":{"images":["8",0],"filename_prefix":"ComfyUI"},"class_type":"SaveImage"},"10":{"inputs":{"vae_name":"ae.sft"},"class_type":"VAELoader"},"11":{"inputs":{"type":"flux","clip_name1":"t5xxl_fp16.safetensors","clip_name2":"clip_l.safetensors"},"class_type":"DualCLIPLoader"},"12":{"inputs":{"unet_name":"flux1-dev.sft","weight_dtype":"default"},"class_type":"UNETLoader"},"13":{"inputs":{"noise":["25",0],"guider":["22",0],"sigmas":["17",0],"sampler":["16",0],"latent_image":["27",0]},"class_type":"SamplerCustomAdvanced"},"16":{"inputs":{"sampler_name":"euler"},"class_type":"KSamplerSelect"},"17":{"inputs":{"model":["30",0],"steps":20,"denoise":1,"scheduler":"simple"},"class_type":"BasicScheduler"},"22":{"inputs":{"model":["30",0],"conditioning":["26",0]},"class_type":"BasicGuider"},"25":{"inputs":{"noise_seed":219670278747233},"class_type":"RandomNoise"},"26":{"inputs":{"guidance":3.5,"conditioning":["6",0]},"class_type":"FluxGuidance"},"27":{"inputs":{"width":1024,"height":1024,"batch_size":1},"class_type":"EmptySD3LatentImage"},"30":{"inputs":{"model":["12",0],"width":1024,"height":1024,"max_shift":1.15,"base_shift":0.5},"class_type":"ModelSamplingFlux"},"38":{"inputs":{"input_id":"positive_prompt","description":"","display_name":"","default_value":"cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open holding a fancy black forest cake with candles on top in the kitchen of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere"},"class_type":"ComfyUIDeployExternalText"}}"""
    
    # workflow_api_raw = """{"3":{"inputs":{"cfg":5.45,"seed":844390075629552,"model":["4",0],"steps":20,"denoise":1,"negative":["40",0],"positive":["16",0],"scheduler":"sgm_uniform","latent_image":["53",0],"sampler_name":"euler"},"class_type":"KSampler"},"4":{"inputs":{"ckpt_name":"sd3.5_large.safetensors"},"class_type":"CheckpointLoaderSimple"},"8":{"inputs":{"vae":["4",2],"samples":["3",0]},"class_type":"VAEDecode"},"9":{"inputs":{"images":["8",0],"filename_prefix":"ComfyUI"},"class_type":"SaveImage"},"16":{"inputs":{"clip":["43",0],"text":["56",0]},"class_type":"CLIPTextEncode"},"40":{"inputs":{"clip":["43",0],"text":""},"class_type":"CLIPTextEncode"},"41":{"inputs":{"type":"sd3","clip_name":"t5xxl_fp16.safetensors"},"class_type":"CLIPLoader"},"42":{"inputs":{"type":"sd3","clip_name1":"clip_l.safetensors","clip_name2":"clip_g.safetensors"},"class_type":"DualCLIPLoader"},"43":{"inputs":{"clip_name1":"clip_l.safetensors","clip_name2":"clip_g.safetensors","clip_name3":"t5xxl_fp16.safetensors"},"class_type":"TripleCLIPLoader"},"53":{"inputs":{"width":1024,"height":1024,"batch_size":1},"class_type":"EmptySD3LatentImage"},"54":{"inputs":{"ckpt_name":"sd3.5_large.safetensors"},"class_type":"CheckpointLoaderSimple"},"56":{"inputs":{"input_id":"positive_prompt","description":"","display_name":"","default_value":"a bottle with a rainbow galaxy inside it on top of a wooden table on a snowy mountain top with the ocean and clouds in the background with a shot glass beside containing darkness beside a snow sculpture in the shape of a fox"},"class_type":"ComfyUIDeployExternalText"}}"""

    models_to_cache = [
        "/comfyui/models/unet/flux1-schnell.sft",
        "/comfyui/models/clip/clip_l.safetensors",
        "/comfyui/models/clip/clip_g.safetensors",
        "/comfyui/models/clip/t5xxl_fp16.safetensors",
        "/comfyui/models/vae/ae.sft",
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
