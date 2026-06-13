from fastapi import APIRouter

from ...config import get_settings
from ...constants import MODEL_NAMES


# Router de salud del backend y estado general de configuración.
router = APIRouter()


@router.get("/health")
def health():
    # Obtiene la configuración activa de la aplicación.
    s = get_settings()

    # Devuelve información básica para verificar que el backend está operativo.
    return {
        "status": "ok",
        "app": s.app_name,
        "env": s.app_env,
        "default_model": s.liveness_default_model,
        "available_models": sorted(MODEL_NAMES),
        "weights_dir": str(s.weights_dir),
    }