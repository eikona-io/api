config = {
    "name": "my-app",
    "deploy_test": "False",
    "gpu": "L4",
    "public_model_volume": "models_org_2bWQ1FoWC3Wro391TurkeVG77pC",
    "private_model_volume": "private-model-store",
    "pip": [],
    "run_timeout": 60 * 5,
    "idle_timeout": 60,
    "gpu_event_callback_url": "",
    "cd_callback_url": "",
    "allow_concurrent_inputs": 50,
    "concurrency_limit": 2,
    "legacy_mode": "False",
    "install_custom_node_with_gpu": "False",
    "ws_timeout": 2,
    "deps": {
        "comfyui": "7faa4507ecbd2ad67afcdea44b46ecdceec75232",
        "custom_nodes": {
            # "https://github.com/ltdrdata/ComfyUI-Impact-Pack": {
            #     "url": "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
            #     "name": "ComfyUI Impact Pack",
            #     "hash": "585787bfa7fe0916821add13aa0e2a01c999a4df",
            #     "warning": "No hash found in snapshot, using latest commit hash",
            #     "pip": [
            #         "ultralytics"
            #     ]
            # }
        },
        "models": {"checkpoints": [{"name": "SD1.5/V07_v07.safetensors"}]},
        "files": {"images": [{"name": "2pass-original.png"}]},
    },
    "auth_token": "",
    "machine_id": "",
    "docker_commands": [
        [
            "FROM base",
            "WORKDIR /comfyui/custom_nodes",
            "RUN git clone https://github.com/BennyKok/comfyui-deploy",
            "WORKDIR /comfyui/custom_nodes/comfyui-deploy",
            "RUN git checkout a640e1eb792eeb25d5b99bdf6cb561ab3b51610b",
            "RUN if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi",
            "RUN if [ -f install.py ]; then python install.py; fi",
        ]
    ],
}
