from fastapi import UploadFile, HTTPException
from ..config import get_settings


# Utilidades de validación para archivos recibidos desde los endpoints.
async def validate_upload(file: UploadFile) -> bytes:
    # Lee el contenido completo del archivo subido.
    data = await file.read()

    # Rechaza archivos vacíos para evitar procesamiento inválido.
    if not data:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    # Rechaza archivos que superen el tamaño máximo configurado.
    if len(data) > get_settings().max_upload_bytes:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande")

    # Devuelve los bytes validados para lectura o inferencia posterior.
    return data