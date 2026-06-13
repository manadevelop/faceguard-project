from fastapi import APIRouter, UploadFile, File, Form
from ...utils.validation_utils import validate_upload
from ...utils.image_utils import read_image_bytes
from ...services.arcface_service import ArcFaceService
from ...config import get_settings


# Router de identidad facial: enrolamiento y verificación por person_id.
router = APIRouter()

# Servicio encargado de generar, guardar y comparar embeddings faciales.
arcface = ArcFaceService()


@router.post("/enroll")
async def enroll(person_id: str = Form(...), face: UploadFile = File(...)):
    # Registra una identidad facial asociada a un person_id.
    data = await validate_upload(face)

    # Convierte la imagen recibida a un formato válido para extracción de embedding.
    img = read_image_bytes(data)

    # Genera y almacena el embedding facial correspondiente al person_id.
    return arcface.enroll(person_id, img)


@router.post("/verify")
async def verify(person_id: str = Form(...), face: UploadFile = File(...)):
    # Verifica si el rostro recibido corresponde al person_id indicado.
    data = await validate_upload(face)

    # Convierte la imagen recibida a un formato válido para comparación facial.
    img = read_image_bytes(data)

    # Compara el embedding del rostro contra el registro usando el threshold configurado.
    return arcface.verify(person_id, img, get_settings().identity_threshold)