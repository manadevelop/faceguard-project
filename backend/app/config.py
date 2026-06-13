# Importa lru_cache para reutilizar la misma configuración sin recrearla varias veces.
from functools import lru_cache
# Importa Path para construir rutas de archivos de forma portable.
from pathlib import Path

# Importa BaseSettings y SettingsConfigDict para leer configuración desde variables de entorno.
from pydantic_settings import BaseSettings, SettingsConfigDict


# Define la configuración central de la aplicación FaceGuard.
class Settings(BaseSettings):
    # Nombre de la aplicación mostrado o usado internamente.
    app_name: str = "FaceGuard"
    # Entorno de ejecución por defecto de la aplicación.
    app_env: str = "local"
    # Modelo anti-spoofing usado por defecto si el usuario no selecciona otro.
    liveness_default_model: str = "efficientnet_b0"
    # Threshold general para decidir si una muestra supera la validación de vida.
    liveness_threshold: float = 0.70
    # Threshold usado para comparar embeddings en verificación de identidad.
    identity_threshold: float = 0.65
    # Tamaño máximo permitido para archivos subidos al backend.
    max_upload_bytes: int = 2_000_000
    # Tamaño esperado del crop facial que ingresa al modelo.
    expected_crop_size: int = 224
    # Ruta local donde se almacenan los checkpoints entrenados de los modelos.
    weights_dir: Path = Path(__file__).resolve().parent / "models" / "weights"

    # Indica que la configuración también puede leerse desde un archivo .env en UTF-8.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Mantiene en caché la configuración para no reconstruirla en cada llamada.
@lru_cache
# Devuelve una instancia única de Settings para toda la aplicación.
def get_settings() -> Settings:
    # Crea y retorna la configuración cargada desde valores por defecto y variables de entorno.
    return Settings()