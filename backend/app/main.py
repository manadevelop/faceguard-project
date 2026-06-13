# Importa Path para construir rutas de archivos de forma portable.
from pathlib import Path

# Importa FastAPI para crear la aplicación backend.
from fastapi import FastAPI
# Importa el middleware CORS para permitir solicitudes desde el frontend.
from fastapi.middleware.cors import CORSMiddleware
# Importa respuestas para servir archivos y redireccionar rutas.
from fastapi.responses import FileResponse, RedirectResponse
# Importa StaticFiles para exponer archivos estáticos del frontend.
from fastapi.staticfiles import StaticFiles

# Importa el router principal que agrupa los endpoints de la API.
from .api.router import api_router
# Importa la configuración de logging del proyecto.
from .logging_config import configure_logging

# Configura el sistema de logs antes de iniciar la aplicación.
configure_logging()
# Crea la instancia principal de FastAPI con título y versión.
app = FastAPI(title="FaceGuard API", version="0.1.0")

# Agrega configuración CORS para permitir comunicación entre frontend y backend.
app.add_middleware(
    CORSMiddleware,
    # Permite solicitudes desde cualquier origen.
    allow_origins=["*"],
    # Permite enviar credenciales si el cliente las usa.
    allow_credentials=True,
    # Permite todos los métodos HTTP.
    allow_methods=["*"],
    # Permite todos los encabezados HTTP.
    allow_headers=["*"],
)

# Registra todos los endpoints de la API bajo el prefijo /api/v1.
app.include_router(api_router, prefix="/api/v1")

# Define la ruta local donde se encuentra la carpeta frontend.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

# Verifica si existe la carpeta frontend antes de montarla.
if FRONTEND_DIR.exists():
    # Monta los archivos estáticos del frontend para servir CSS, JS e imágenes.
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# Define la ruta raíz de la aplicación.
@app.get("/", include_in_schema=False)
def root():
    """Redirige al contexto principal de la aplicación web."""
    # Redirige al usuario hacia la demo web principal.
    return RedirectResponse(url="/faceproguard")


# Define la ruta principal donde se sirve la aplicación web FaceGuard.
@app.get("/faceproguard", include_in_schema=False)
def faceproguard_index():
    """Context path de la demo web local."""
    # Construye la ruta al archivo index.html del frontend.
    index_file = FRONTEND_DIR / "index.html"
    # Si index.html existe, lo devuelve como respuesta HTML.
    if index_file.exists():
        return FileResponse(index_file)
    # Si no existe el frontend, devuelve un mensaje indicando que solo la API está activa.
    return {"message": "FaceGuard API running", "frontend": "not found"}


# Define la misma ruta con slash final para evitar errores de navegación.
@app.get("/faceproguard/", include_in_schema=False)
def faceproguard_index_slash():
    # Reutiliza la misma respuesta de la ruta /faceproguard.
    return faceproguard_index()