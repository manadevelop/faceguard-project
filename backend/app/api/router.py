from fastapi import APIRouter
from .v1 import health, liveness, identity, auth

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(liveness.router, prefix="/liveness", tags=["liveness"])
api_router.include_router(identity.router, prefix="/identity", tags=["identity"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
