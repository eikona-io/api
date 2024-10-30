import json
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from deployed.comfy_utils import Input
import modal
from pydantic import BaseModel, validator

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])


diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

flux_image = cuda_dev_image.apt_install(
    "git",
    "libglib2.0-0",
    "libsm6",
    "libxrender1",
    "libxext6",
    "ffmpeg",
    "libgl1",
).pip_install(
    "invisible_watermark==0.2.0",
    "transformers==4.44.0",
    "accelerate==0.33.0",
    "safetensors==0.4.4",
    "sentencepiece==0.2.0",
    "torch==2.5.0",
    f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
    "numpy<2",
)

flux_image = flux_image.env({"TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache"}).env(
    {"TORCHINDUCTOR_FX_GRAPH_CACHE": "1"}
)

app = modal.App("sd3-5", image=flux_image)

with flux_image.imports():
    import torch
    from diffusers import StableDiffusion3Pipeline

MINUTES = 60  # seconds
NUM_INFERENCE_STEPS = 40  # use ~50 for [dev], smaller for [schnell]
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
SAFE_MODEL_NAME = MODEL_ID.replace("/", "-")  # Replace slashes with dashes
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
    "/models-cache": modal.Volume.lookup("models_org_2am4LjkQ5IaWGRYMHxGXfHdHcjA"),
}


@app.cls(
    gpu="H100",  # fastest GPU on Modal
    container_idle_timeout=5 * MINUTES,
    timeout=60 * MINUTES,  # leave plenty of time for compilation
    secrets=[modal.Secret.from_name("hf-models-download")],
    volumes=volumes,
)
class ComfyDeployRunner:
    compile: int = (  # see section on torch.compile below for details
        modal.parameter(default=0)
    )

    def setup_model(self):
        from huggingface_hub import snapshot_download
        from transformers.utils import move_cache

        snapshot_download(MODEL_ID)

        move_cache()

        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        )

        return pipe

    @modal.build()
    def build(self):
        self.setup_model()

    @modal.enter()
    def enter(self):
        pipe = self.setup_model()
        pipe.to("cuda")  # move model to GPU
        self.pipe = optimize(pipe, compile=bool(self.compile))

    @modal.method()
    def run(self, input: Input) -> bytes:
        if isinstance(input, dict):
            input = Input(**input)

        print("🎨 generating image...")
        out = self.pipe(
            input.inputs.get("positive_prompt"),
            output_type="pil",
            num_inference_steps=NUM_INFERENCE_STEPS,
        ).images[0]

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()


@app.local_entrypoint()
def main(
    prompt: str = "a computer screen showing ASCII terminal art of the"
    " word 'Modal' in neon green. two programmers are pointing excitedly"
    " at the screen.",
    twice: bool = True,
    compile: bool = False,
):
    t0 = time.time()
    image_bytes = ComfyDeployRunner(compile=compile).run.remote(prompt)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = ComfyDeployRunner(compile=compile).run.remote(prompt)
        print(f"🎨 second inference latency: {time.time() - t0:.2f} seconds")

    output_path = Path("/tmp") / "flux" / "output.jpg"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"🎨 saving output to {output_path}")
    output_path.write_bytes(image_bytes)


def optimize(pipe, compile=True):
    # fuse QKV projections in Transformer and VAE
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # set torch compile flags
    config = torch._inductor.config
    config.disable_progress = False  # show progress bar
    config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    # adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

    # tag the compute-intensive modules, the Transformer and VAE decoder, for compilation
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    # trigger torch compilation
    print("🔦 running torch compiliation (may take up to 20 minutes)...")

    pipe(
        "dummy prompt to trigger torch compilation",
        output_type="pil",
        num_inference_steps=NUM_INFERENCE_STEPS,  # use ~50 for [dev], smaller for [schnell]
    ).images[0]

    print("🔦 finished torch compilation")

    return pipe
