import modal


SHARED_CLS_CONFIG = {
    "container_idle_timeout": 300,
    "timeout": 60 * 60,  # leave plenty of time for compilation
    "gpu": "H100",
    "secrets": [modal.Secret.from_name("hf-models-download")],
}


def get_volumes(safe_model_name: str):
    volumes = {  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
        "/root/.nv": modal.Volume.from_name(
            "nv-cache-" + safe_model_name, create_if_missing=True
        ),
        "/root/.triton": modal.Volume.from_name(
            "triton-cache-" + safe_model_name, create_if_missing=True
        ),
        "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache-" + safe_model_name, create_if_missing=True
        ),
        # "/models-cache": modal.Volume.lookup("models_org_2am4LjkQ5IaWGRYMHxGXfHdHcjA"),
        "/private_models": modal.Volume.lookup(
            "models_org_2am4LjkQ5IaWGRYMHxGXfHdHcjA"
        ),
    }
    return volumes
