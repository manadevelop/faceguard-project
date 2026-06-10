from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List

from ...utils.validation_utils import validate_upload
from ...utils.image_utils import read_image_bytes
from ...services.antispoof_service import AntispoofService
from ...exceptions import FaceGuardError

router = APIRouter()
service = AntispoofService()


@router.get("/models")
def list_liveness_models():
    """Lista modelos anti-spoofing soportados y checkpoints disponibles."""
    return {"models": service.list_models()}


@router.post("/image")
async def verify_image(face: UploadFile = File(...), model: str | None = Query(default=None)):
    try:
        data = await validate_upload(face)
        img = read_image_bytes(data)
        return service.predict_image(img, model)
    except FaceGuardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/frames")
async def verify_frames(frames: List[UploadFile] = File(...), model: str | None = Query(default=None)):
    try:
        imgs = []
        for f in frames:
            data = await validate_upload(f)
            imgs.append(read_image_bytes(data))
        return service.predict_frames(imgs, model)
    except FaceGuardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
