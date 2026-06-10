from fastapi import APIRouter, UploadFile, File, Form
from ...utils.validation_utils import validate_upload
from ...utils.image_utils import read_image_bytes
from ...services.arcface_service import ArcFaceService
from ...config import get_settings

router = APIRouter()
arcface = ArcFaceService()

@router.post("/enroll")
async def enroll(person_id: str = Form(...), face: UploadFile = File(...)):
    data = await validate_upload(face)
    img = read_image_bytes(data)
    return arcface.enroll(person_id, img)

@router.post("/verify")
async def verify(person_id: str = Form(...), face: UploadFile = File(...)):
    data = await validate_upload(face)
    img = read_image_bytes(data)
    return arcface.verify(person_id, img, get_settings().identity_threshold)
