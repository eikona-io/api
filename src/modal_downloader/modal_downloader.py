import json
from typing import Optional
import modal
import logging

modal_downloader_app = modal.App("volume-operations")

@modal_downloader_app.function(
    timeout=3600,
    secrets=[modal.Secret.from_name("civitai-api-key")],
    image=modal.Image.debian_slim().pip_install("aiohttp", "huggingface_hub"),
)
async def modal_download_file_task(
    download_url,
    folder_path,
    filename,
    db_model_id,
    full_path,
    volume_name,
    upload_type,
    token,
):
    import time
    import os
    from enum import Enum
    from modal import Volume
    import aiohttp
    from huggingface_hub import hf_hub_download
    import re
    import asyncio

    print("download_file_task start")

    class ModelDownloadStatus(Enum):
        PROGRESS = "progress"
        SUCCESS = "success"
        FAILED = "failed"

    def create_status_payload(
        model_id,
        progress,
        status: ModelDownloadStatus,
        error_log: str = None,
    ):
        payload = {
            "model_id": model_id,
            "download_progress": progress,
            "status": status.value,
        }
        if error_log:
            payload["error_log"] = error_log
        return payload

    def extract_huggingface_repo_id(url: str) -> str | None:
        match = re.search(r"huggingface\.co/([^/]+/[^/]+)", url)
        return match.group(1) if match else None

    async def download_url_file(download_url: str, token: Optional[str]):
        # Create a shared state for progress tracking
        progress_state = {
            "downloaded_size": 0,
            "total_size": 0,
            "should_stop": False
        }

        async def report_progress():
            last_update_time = time.time()
            while not progress_state["should_stop"]:
                current_time = time.time()
                if current_time - last_update_time >= 5:
                    progress = (
                        int((progress_state["downloaded_size"] / progress_state["total_size"]) * 90)
                        if progress_state["total_size"]
                        else 0
                    )
                    logging.info(f"Download progress: {progress}% ({progress_state['downloaded_size']}/{progress_state['total_size']} bytes)")
                    progress_state["last_status"] = create_status_payload(
                        db_model_id,
                        progress,
                        ModelDownloadStatus.PROGRESS,
                    )
                    last_update_time = current_time
                await asyncio.sleep(1)  # Check progress every second

        try:
            headers = {
                "Accept-Encoding": "identity",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3600)
            ) as session:
                async with session.get(download_url, headers=headers) as response:
                    response.raise_for_status()
                    progress_state["total_size"] = int(response.headers.get("Content-Length", 0))
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)

                    # Start progress reporting task
                    progress_task = asyncio.create_task(report_progress())

                    with open(full_path, "wb") as file:
                        async for data in response.content.iter_chunked(65536):
                            if data:
                                file.write(data)
                                progress_state["downloaded_size"] += len(data)
                                if "last_status" in progress_state:
                                    yield progress_state["last_status"]
                                    del progress_state["last_status"]

                    # Clean up progress reporting
                    progress_state["should_stop"] = True
                    await progress_task

        except Exception as e:
            progress_state["should_stop"] = True  # Ensure progress reporting stops
            yield create_status_payload(
                db_model_id, 0, ModelDownloadStatus.FAILED, str(e)
            )
            raise e

    try:
        downloaded_path = None
        if upload_type == "huggingface":
            async for event in download_url_file(download_url, token):
                yield event
            downloaded_path = full_path
        elif upload_type == "download-url":
            async for event in download_url_file(download_url, None):
                yield event
            downloaded_path = full_path
        elif upload_type == "civitai":
            if "civitai.com" in download_url:
                download_url += f"{'&' if '?' in download_url else '?'}token={os.environ['CIVITAI_KEY']}"
            async for event in download_url_file(download_url, None):
                yield event
            downloaded_path = full_path
        else:
            raise ValueError(f"Unsupported upload_type: {upload_type}")

        volume = Volume.lookup(volume_name, create_if_missing=True)
        
        with volume.batch_upload() as batch:
            batch.put_file(downloaded_path, full_path)

        yield create_status_payload(
            db_model_id, 100, ModelDownloadStatus.SUCCESS
        )
    except Exception as e:
        print(f"Error in download_file_task: {str(e)}")
        yield create_status_payload(
            db_model_id, 0, ModelDownloadStatus.FAILED, str(e)
        )
        raise e
