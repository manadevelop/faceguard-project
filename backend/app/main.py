from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api.router import api_router
from .logging_config import configure_logging

configure_logging()
app = FastAPI(title="FaceGuard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

if FRONTEND_DIR.exists():
    # Se mantiene esta ruta para cargar CSS/JS de forma directa.
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/", include_in_schema=False)
def root():
    """Redirige al contexto principal de la aplicación web."""
    return RedirectResponse(url="/faceproguard")


@app.get("/faceproguard", include_in_schema=False)
def faceproguard_index():
    """Context path de la demo web local."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "FaceGuard API running", "frontend": "not found"}


@app.get("/faceproguard/", include_in_schema=False)
def faceproguard_index_slash():
    return faceproguard_index()
