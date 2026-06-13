from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from typing import List

from ...utils.validation_utils import validate_upload
from ...utils.image_utils import read_image_bytes
from ...services.antispoof_service import AntispoofService
from ...services.arcface_service import ArcFaceService
from ...services.decision_service import DecisionService
from ...config import get_settings
from ...exceptions import FaceGuardError


# Router de autenticación facial: integra liveness, identidad y decisión final.
router = APIRouter()

# Servicio encargado de ejecutar la detección de vida anti-spoofing.
antispoof = AntispoofService()

# Servicio encargado de enrolar y verificar identidad facial mediante embeddings.
arcface = ArcFaceService()

# Servicio encargado de combinar liveness e identidad para decidir acceso.
decision_service = DecisionService()


@router.post("/verify-image")
async def verify_image(
    face: UploadFile = File(...),
    person_id: str | None = Form(default=None),
    model: str | None = Query(default=None),
):
    # Procesa una imagen facial, valida liveness y opcionalmente verifica identidad.
    try:
        # Valida el archivo recibido y obtiene sus bytes.
        data = await validate_upload(face)

        # Convierte los bytes de la imagen a un objeto procesable por el modelo.
        img = read_image_bytes(data)

        # Ejecuta el modelo anti-spoofing sobre la imagen recibida.
        liveness = antispoof.predict_image(img, model)

        # Inicializa la verificación de identidad como nula.
        identity = None

        # Verifica identidad solo si la muestra es viva y se envió person_id.
        if liveness["is_live"] and person_id:
            identity = arcface.verify(person_id, img, get_settings().identity_threshold)

        # Combina el resultado de vida e identidad para decidir acceso.
        decision = decision_service.decide(liveness, identity)

        # Devuelve la decisión final junto con los detalles de liveness e identidad.
        return {**decision, "liveness": liveness, "identity": identity}

    # Devuelve error 400 para errores controlados del dominio FaceGuard.
    except FaceGuardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Devuelve error 500 si falta algún archivo requerido, como checkpoints.
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Devuelve error 500 ante fallos de ejecución del modelo o backend.
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/verify-realtime")
async def verify_realtime(
    frames: List[UploadFile] = File(...),
    person_id: str | None = Form(default=None),
    model: str | None = Query(default=None),
):
    # Procesa múltiples frames para validación de vida en modo tiempo real.
    try:
        # Lista donde se almacenan los frames decodificados.
        imgs = []

        # Valida y decodifica cada frame recibido desde el frontend.
        for f in frames:
            data = await validate_upload(f)
            imgs.append(read_image_bytes(data))

        # Ejecuta la inferencia anti-spoofing sobre la secuencia de frames.
        liveness = antispoof.predict_frames(imgs, model)

        # Inicializa la verificación de identidad como nula.
        identity = None

        # Verifica identidad usando el último frame si la muestra fue considerada viva.
        if liveness["is_live"] and person_id:
            identity = arcface.verify(person_id, imgs[-1], get_settings().identity_threshold)

        # Combina el resultado de vida e identidad para decidir acceso.
        decision = decision_service.decide(liveness, identity)

        # Devuelve la decisión final junto con liveness e identidad.
        return {**decision, "liveness": liveness, "identity": identity}

    # Devuelve error 400 para errores controlados del dominio FaceGuard.
    except FaceGuardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Devuelve error 500 si falta algún archivo requerido, como checkpoints.
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Devuelve error 500 ante fallos de ejecución del modelo o backend.
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc