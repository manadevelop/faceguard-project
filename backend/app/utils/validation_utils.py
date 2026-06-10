from fastapi import UploadFile, HTTPException
from ..config import get_settings

async def validate_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Archivo vacío")
    if len(data) > get_settings().max_upload_bytes:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande")
    return data
