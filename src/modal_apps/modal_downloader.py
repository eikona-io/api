from typing import Optional, AsyncGenerator, Dict, Optional, List
import logging
import time
import os
from enum import Enum
from modal import Volume, Image, App, Secret
import aiohttp
import asyncio
import tempfile
from huggingface_hub import snapshot_download, HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

modal_downloader_app = App("volume-operations")

image = Image.debian_slim().pip_install(
    "huggingface_hub>=0.19.4",
    "tqdm>=4.66.1",
    "aiohttp>=3.8.6",
)

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


@modal_downloader_app.function(
    timeout=3600,
    secrets=[Secret.from_name("civitai-api-key")],
    image=image
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
    print("download_file_task start")

    print(f"download_url: {download_url}")
    print(f"folder_path: {folder_path}")
    print(f"filename: {filename}")
    print(f"db_model_id: {db_model_id}")
    print(f"full_path: {full_path}")
    print(f"volume_name: {volume_name}")
    print(f"upload_type: {upload_type}")

    async def download_url_file(download_url: str, token: Optional[str]):
        # Create a shared state for progress tracking
        progress_state = {"downloaded_size": 0, "total_size": 0, "should_stop": False}

        async def report_progress():
            last_update_time = time.time()
            while not progress_state["should_stop"]:
                current_time = time.time()
                if current_time - last_update_time >= 5:
                    progress = (
                        int(
                            (
                                progress_state["downloaded_size"]
                                / progress_state["total_size"]
                            )
                            * 90
                        )
                        if progress_state["total_size"]
                        else 0
                    )
                    logging.info(
                        f"Download progress: {progress}% ({progress_state['downloaded_size']}/{progress_state['total_size']} bytes)"
                    )
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
                    progress_state["total_size"] = int(
                        response.headers.get("Content-Length", 0)
                    )
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

        yield create_status_payload(db_model_id, 100, ModelDownloadStatus.SUCCESS)
    except Exception as e:
        print(f"Error in download_file_task: {str(e)}")
        yield create_status_payload(db_model_id, 0, ModelDownloadStatus.FAILED, str(e))
        raise e

async def report_progress(
    progress: float, 
    model_id: str, 
    status: str = "progress", 
    error_log: Optional[str] = None
) -> Dict:
    """Generate progress report events"""
    return {
        "status": status,
        "download_progress": int(progress),
        "model_id": model_id,
        "error_log": error_log,
    }

@modal_downloader_app.function(timeout=3600, image=image)
async def modal_download_repo_task(
    repo_id: str,
    folder_path: str,
    model_id: str,
    volume_name: str,
    token: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Download an entire HuggingFace repository to a Modal volume.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "Skywork/SkyReels-A1")
        folder_path: Path within the volume to store the repository
        model_id: Database ID of the model record
        volume_name: Name of the Modal volume
        token: HuggingFace API token (optional)
    
    Yields:
        Progress updates as dictionaries
    """
    try:
        # Use user token if provided, otherwise fall back to environment variable
        if not token:
            token = os.environ.get("HUGGINGFACE_TOKEN")
        
        # Get the Modal volume
        volume = Volume.from_name(volume_name)
        
        # Create a temporary directory for downloading
        with tempfile.TemporaryDirectory() as temp_dir:
            # Report initial progress
            yield await report_progress(0, model_id)
            
            try:
                # Download the repository
                local_dir = snapshot_download(
                    repo_id=repo_id,
                    local_dir=temp_dir,
                    token=token,
                    local_dir_use_symlinks=False,
                )
                
                # Get list of files
                files = list(Path(local_dir).glob("**/*"))
                total_files = len(files)
                
                if total_files == 0:
                    yield await report_progress(
                        0, 
                        model_id, 
                        status="failed", 
                        error_log="Repository is empty"
                    )
                    return
                
                # Create destination directory in volume if it doesn't exist
                dest_dir = os.path.join(folder_path, repo_id.split("/")[-1])
                try:
                    volume.mkdir(dest_dir, exist_ok=True, parents=True)
                except Exception as e:
                    yield await report_progress(
                        0, 
                        model_id, 
                        status="failed", 
                        error_log=f"Failed to create directory: {str(e)}"
                    )
                    return
                
                # Copy files to volume
                for i, file_path in enumerate(files, 1):
                    if file_path.is_file():
                        # Calculate relative path
                        rel_path = file_path.relative_to(local_dir)
                        dest_path = os.path.join(dest_dir, str(rel_path))
                        
                        # Create parent directories if needed
                        parent_dir = os.path.dirname(dest_path)
                        if parent_dir:
                            volume.mkdir(parent_dir, exist_ok=True, parents=True)
                        
                        # Copy file to volume
                        with open(file_path, "rb") as f:
                            volume.write_bytes(dest_path, f.read())
                        
                        # Report progress
                        progress = min(int((i / total_files) * 100), 99)
                        yield await report_progress(progress, model_id)
                
                # Report success
                yield await report_progress(100, model_id, status="success")
                
            except RepositoryNotFoundError:
                yield await report_progress(
                    0, 
                    model_id, 
                    status="failed", 
                    error_log=f"Repository {repo_id} not found"
                )
            except RevisionNotFoundError:
                yield await report_progress(
                    0, 
                    model_id, 
                    status="failed", 
                    error_log=f"Revision not found for repository {repo_id}"
                )
            except Exception as e:
                yield await report_progress(
                    0, 
                    model_id, 
                    status="failed", 
                    error_log=f"Error downloading repository: {str(e)}"
                )
    
    except Exception as e:
        yield await report_progress(
            0, 
            model_id, 
            status="failed", 
            error_log=f"Unexpected error: {str(e)}"
        )