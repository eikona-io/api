from pathlib import Path
from typing import Annotated
import uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import modal

modal_file_uploader_app = modal.App("volume-file-uploader")

web_app = FastAPI()

@web_app.post("/")
async def upload(
    file: UploadFile = File(),
    volume_name: str = Form(),
    filename: str = Form(),
    target_path: str = Form(default="/input"),
    subfolder: str = Form(default=None),
):
    # Lookup the volume
    from tempfile import gettempdir

    # Create a unique file path to avoid conflicts and overwrites
    local_location = Path(gettempdir()) / f"{uuid.uuid4()}_{filename}"
    file_location = Path(target_path) / f"{filename}"
    if subfolder is not None:
        file_location = Path(target_path) / subfolder / f"{filename}"
    print(local_location, file_location, volume_name)

    volume = await modal.Volume.lookup.aio(volume_name, create_if_missing=True)

    try:
        # Write file to local storage
        with local_location.open("wb") as buffer:
            data = await file.read()
            buffer.write(data)

        # Upload file to the volume
        async with await volume.batch_upload.aio() as batch:
            batch.put_file(str(local_location), str(file_location))

        # Delete the local file after uploading
        local_location.unlink()

    except Exception as e:
        # Provide a more specific error message
        raise HTTPException(
            status_code=500, detail=f"Failed to upload file: {str(e)}"
        )

    return {"message": "File uploaded successfully", "path": str(file_location)}

@modal_file_uploader_app.function(
    timeout=3600,
    # secrets=[modal.Secret.from_name("civitai-api-key")],
    image=modal.Image.debian_slim().pip_install("aiohttp", "python-multipart"),
)
@modal.asgi_app(
    label="volume-file-uploader-upload",
)
def fastapi_app():
    return web_app