from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from typing import List

from ...utils.validation_utils import validate_upload
from ...utils.image_utils import read_image_bytes
from ...services.antispoof_service import AntispoofService
from ...services.arcface_service import ArcFaceService
from ...services.decision_service import DecisionService
from ...config import get_settings
from ...exceptions import FaceGuardError

router = APIRouter()
antispoof = AntispoofService()
arcface = ArcFaceService()
decision_service = DecisionService()


@router.post("/verify-image")
async def verify_image(
    face: UploadFile = File(...),
    person_id: str | None = Form(default=None),
    model: str | None = Query(default=None),
):
    try:
        data = await validate_upload(face)
        img = read_image_bytes(data)
        liveness = antispoof.predict_image(img, model)
        identity = None
        if liveness["is_live"] and person_id:
            identity = arcface.verify(person_id, img, get_settings().identity_threshold)
        decision = decision_service.decide(liveness, identity)
        return {**decision, "liveness": liveness, "identity": identity}
    except FaceGuardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/verify-realtime")
async def verify_realtime(
    frames: List[UploadFile] = File(...),
    person_id: str | None = Form(default=None),
    model: str | None = Query(default=None),
):
    try:
        imgs = []
        for f in frames:
            data = await validate_upload(f)
            imgs.append(read_image_bytes(data))
        liveness = antispoof.predict_frames(imgs, model)
        identity = None
        if liveness["is_live"] and person_id:
            identity = arcface.verify(person_id, imgs[-1], get_settings().identity_threshold)
        decision = decision_service.decide(liveness, identity)
        return {**decision, "liveness": liveness, "identity": identity}
    except FaceGuardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
