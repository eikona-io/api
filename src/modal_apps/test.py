import uuid
import modal
import asyncio
import time

workflow_api = """{"3":{"inputs":{"cfg":8,"seed":156680208700286,"model":["4",0],"steps":20,"denoise":1,"negative":["7",0],"positive":["6",0],"scheduler":"normal","latent_image":["5",0],"sampler_name":"euler"},"class_type":"KSampler"},"4":{"inputs":{"ckpt_name":"v1-5-pruned-emaonly.ckpt"},"class_type":"CheckpointLoaderSimple"},"5":{"inputs":{"width":512,"height":512,"batch_size":1},"class_type":"EmptyLatentImage"},"6":{"inputs":{"clip":["4",1],"text":["12",0]},"class_type":"CLIPTextEncode"},"7":{"inputs":{"clip":["4",1],"text":["13",0]},"class_type":"CLIPTextEncode"},"8":{"inputs":{"vae":["4",2],"samples":["3",0]},"class_type":"VAEDecode"},"9":{"inputs":{"images":["8",0],"filename_prefix":"ComfyUI"},"class_type":"SaveImage"},"12":{"inputs":{"input_id":"positive_prompt","default_value":"beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"},"class_type":"ComfyUIDeployExternalText"},"13":{"inputs":{"input_id":"negative_prompt","default_value":"text, watermark"},"class_type":"ComfyUIDeployExternalText"}}"""
flux_workflow_api = """{"6":{"inputs":{"clip":["11",0],"text":["38",0]},"class_type":"CLIPTextEncode"},"8":{"inputs":{"vae":["10",0],"samples":["13",0]},"class_type":"VAEDecode"},"9":{"inputs":{"images":["8",0],"filename_prefix":"ComfyUI"},"class_type":"SaveImage"},"10":{"inputs":{"vae_name":"ae.sft"},"class_type":"VAELoader"},"11":{"inputs":{"type":"flux","clip_name1":"t5xxl_fp16.safetensors","clip_name2":"clip_l.safetensors"},"class_type":"DualCLIPLoader"},"12":{"inputs":{"unet_name":"flux1-dev.sft","weight_dtype":"default"},"class_type":"UNETLoader"},"13":{"inputs":{"noise":["25",0],"guider":["22",0],"sigmas":["17",0],"sampler":["16",0],"latent_image":["27",0]},"class_type":"SamplerCustomAdvanced"},"16":{"inputs":{"sampler_name":"euler"},"class_type":"KSamplerSelect"},"17":{"inputs":{"model":["30",0],"steps":20,"denoise":1,"scheduler":"simple"},"class_type":"BasicScheduler"},"22":{"inputs":{"model":["30",0],"conditioning":["26",0]},"class_type":"BasicGuider"},"25":{"inputs":{"noise_seed":219670278747233},"class_type":"RandomNoise"},"26":{"inputs":{"guidance":3.5,"conditioning":["6",0]},"class_type":"FluxGuidance"},"27":{"inputs":{"width":1024,"height":1024,"batch_size":1},"class_type":"EmptySD3LatentImage"},"30":{"inputs":{"model":["12",0],"width":1024,"height":1024,"max_shift":1.15,"base_shift":0.5},"class_type":"ModelSamplingFlux"},"38":{"inputs":{"input_id":"positive_prompt","description":"","display_name":"","default_value":"cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open holding a fancy black forest cake with candles on top in the kitchen of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere"},"class_type":"ComfyUIDeployExternalText"}}"""


# Define parameter generators
def comfy_params(workflow):
    return {
        "workflow_api_raw": workflow,
        "prompt_id": str(uuid.uuid4()),
        "inputs": {
            "positive_prompt": "beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"
        },
    }


def flux_params(_):
     return {
        "prompt_id": str(uuid.uuid4()),
        "inputs": {
            "positive_prompt": "beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"
        },
    }


# Define runner configurations
comfy_runners = {
    # "ComfyDeployRunner": {
    #     "runner": modal.Cls.lookup("anycomfyui", "ComfyDeployRunner"),
    #     "params": comfy_params
    # },
    # "ComfyDeployRunnerNative": {
    #     "runner": modal.Cls.lookup("anycomfyui", "ComfyDeployRunnerNative"),
    #     "params": comfy_params
    # },
    # "ComfyDeployRunnerOptimizedImports": {
    #     "runner": modal.Cls.lookup("anycomfyui", "ComfyDeployRunnerOptimizedImports"),
    #     "params": comfy_params
    # },
    # "ComfyDeployRunnerOptimizedModels": {
    #     "runner": modal.Cls.lookup("anycomfyui", "ComfyDeployRunnerOptimizedModels"),
    #     "params": comfy_params
    # },
}

standalone_runners = {
    # "Flux Dev": {
    #     "runner": modal.Cls.lookup("flux-dev", "Model"),
    #     "params": flux_params
    # },
    # "Flux Schnell": {
    #     "runner": modal.Cls.lookup("flux-schnell", "Model"),
    #     "params": flux_params
    # },
    "SD 3.5": {
        "runner": modal.Cls.lookup("sd3-5", "ComfyDeployRunner"),
        "params": flux_params,
    },
}


async def timed_run(runner, params):
    start_time = time.time()
    result = await runner.run.remote.aio(params)
    end_time = time.time()
    return end_time - start_time, result


async def run_parallel(runners_dict, workflow=None):
    tasks = []

    for name, runner_info in runners_dict.items():
        params = runner_info["params"](workflow)
        tasks.append(timed_run(runner_info["runner"](), params))

    if not tasks:
        return 0, []

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    return total_time, results


def print_loading_times(title, loading_times):
    if not loading_times:  # Skip if no loading times are available
        return
    print(f"\n{title}:")
    print("-" * 50)
    print(f"{'Step':<35} | {'Time (seconds)':<15}")
    print("-" * 50)
    for step, t in loading_times.items():
        print(f"{step:<35} | {t:<15.2f}")
    print("-" * 50)
    print()


def main():
    # Run workflow-based runners
    print("\n=== Running Workflow-based Models ===")
    total_time, results = asyncio.run(run_parallel(comfy_runners, workflow_api))
    for name, (_time, result) in zip(comfy_runners.keys(), results):
        print_loading_times(
            f"{name} Loading Times {_time:.2f} seconds", result.get("loading_time", {})
        )
    print(f"Total time for workflow runners: {total_time:.2f} seconds")

    time.sleep(5)

    # Run standalone runners
    print("\n=== Running Standalone Models ===")
    total_time, results = asyncio.run(run_parallel(standalone_runners))
    for name, (_time, result) in zip(standalone_runners.keys(), results):
        print_loading_times(f"{name} Loading Times {_time:.2f} seconds", None)
    print(f"Total time for standalone runners: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
