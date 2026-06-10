from fastapi import APIRouter

from ...config import get_settings
from ...constants import MODEL_NAMES

router = APIRouter()


@router.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "app": s.app_name,
        "env": s.app_env,
        "default_model": s.liveness_default_model,
        "available_models": sorted(MODEL_NAMES),
        "weights_dir": str(s.weights_dir),
    }
