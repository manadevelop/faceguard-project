from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List

from ...utils.validation_utils import validate_upload
from ...utils.image_utils import read_image_bytes
from ...services.antispoof_service import AntispoofService
from ...exceptions import FaceGuardError


# Router de detección de vida facial anti-spoofing.
router = APIRouter()

# Servicio encargado de cargar modelos y ejecutar inferencia LIVE/SPOOF.
service = AntispoofService()


@router.get("/models")
def list_liveness_models():
    """Lista modelos anti-spoofing soportados y checkpoints disponibles."""
    # Devuelve los modelos disponibles y su estado de checkpoints.
    return {"models": service.list_models()}


@router.post("/image")
async def verify_image(face: UploadFile = File(...), model: str | None = Query(default=None)):
    # Evalúa una imagen facial individual para decidir si es REAL/LIVE o SPOOF.
    try:
        # Valida el archivo subido desde el cliente.
        data = await validate_upload(face)

        # Decodifica la imagen para que pueda ser procesada por el modelo.
        img = read_image_bytes(data)

        # Ejecuta inferencia anti-spoofing sobre la imagen.
        return service.predict_image(img, model)

    # Devuelve error 400 para errores controlados de validación o modelo no soportado.
    except FaceGuardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Devuelve error 500 si falta algún recurso requerido, como pesos del modelo.
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Devuelve error 500 ante fallos de ejecución durante la inferencia.
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/frames")
async def verify_frames(frames: List[UploadFile] = File(...), model: str | None = Query(default=None)):
    # Evalúa una secuencia de frames para detección de vida en modo video.
    try:
        # Lista donde se almacenarán los frames decodificados.
        imgs = []

        # Valida y convierte cada frame recibido.
        for f in frames:
            data = await validate_upload(f)
            imgs.append(read_image_bytes(data))

        # Ejecuta inferencia anti-spoofing sobre la secuencia de frames.
        return service.predict_frames(imgs, model)

    # Devuelve error 400 para errores controlados de validación o modelo no soportado.
    except FaceGuardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Devuelve error 500 si falta algún recurso requerido, como pesos del modelo.
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Devuelve error 500 ante fallos de ejecución durante la inferencia.
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc