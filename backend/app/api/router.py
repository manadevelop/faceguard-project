from fastapi import APIRouter
from .v1 import health, liveness, identity, auth


# Router principal de la API v1 de FaceGuard.
api_router = APIRouter()

# Registra los endpoints de salud del backend.
api_router.include_router(health.router, tags=["health"])

# Registra los endpoints de detección de vida bajo /liveness.
api_router.include_router(liveness.router, prefix="/liveness", tags=["liveness"])

# Registra los endpoints de enrolamiento y verificación de identidad bajo /identity.
api_router.include_router(identity.router, prefix="/identity", tags=["identity"])

# Registra los endpoints de autenticación facial bajo /auth.
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])