from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FaceGuard"
    app_env: str = "local"
    liveness_default_model: str = "efficientnet_b0"
    liveness_threshold: float = 0.70
    identity_threshold: float = 0.65
    max_upload_bytes: int = 2_000_000
    expected_crop_size: int = 224
    weights_dir: Path = Path(__file__).resolve().parent / "models" / "weights"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
