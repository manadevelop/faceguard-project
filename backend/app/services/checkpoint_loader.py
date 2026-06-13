from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ..config import get_settings
from ..exceptions import UnsupportedModelError
from ..models.model_factory import build_model, get_expected_weight_names, SUPPORTED_MODEL_NAMES


# Estructura que agrupa el modelo cargado y sus metadatos de inferencia.
@dataclass(frozen=True)
class LoadedAntispoofModel:
    model_name: str
    modality: str
    threshold: float
    checkpoint_path: Path
    checkpoint_epoch: int | None
    val_metrics: dict[str, Any]
    model: torch.nn.Module
    device: torch.device


# Selecciona el mejor dispositivo disponible para ejecutar inferencia.
def get_inference_device() -> torch.device:
    # Usa CUDA si hay una GPU NVIDIA disponible.
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Usa MPS si está disponible en equipos Apple Silicon.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # Usa CPU como alternativa por defecto.
    return torch.device("cpu")


# Resuelve la ruta del checkpoint esperado para un modelo y modalidad.
def resolve_checkpoint_path(model_name: str, modality: str = "rgb") -> Path:
    # Obtiene la configuración global de la aplicación.
    settings = get_settings()

    # Obtiene la carpeta donde deben estar los pesos de los modelos.
    weights_dir = Path(settings.weights_dir)

    # Revisa cada nombre de checkpoint esperado hasta encontrar uno existente.
    for candidate in get_expected_weight_names(model_name, modality):
        path = weights_dir / candidate
        if path.exists() and path.is_file():
            return path

    # Construye el mensaje con las rutas esperadas si no se encontró checkpoint.
    expected = ", ".join(str(weights_dir / name) for name in get_expected_weight_names(model_name, modality))

    # Lanza error indicando dónde debe ubicarse el archivo .pt.
    raise FileNotFoundError(
        f"No se encontró checkpoint para model={model_name}, modality={modality}. "
        f"Ubica el .pt en una de estas rutas: {expected}"
    )


# Carga y cachea el modelo anti-spoofing para evitar recargarlo en cada solicitud.
@lru_cache(maxsize=16)
def load_antispoof_model(model_name: str, modality: str = "rgb") -> LoadedAntispoofModel:
    # Normaliza el nombre del modelo recibido.
    model_name = model_name.lower().strip()

    # Normaliza la modalidad solicitada.
    modality = modality.lower().strip()

    # Valida que el modelo esté registrado como soportado.
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise UnsupportedModelError(f"Modelo no soportado: {model_name}")

    # Restringe la inferencia web a RGB porque la webcam no entrega depth.
    if modality != "rgb":
        raise UnsupportedModelError("El backend web solo acepta inferencia RGB desde webcam.")

    # Localiza el checkpoint del modelo solicitado.
    checkpoint_path = resolve_checkpoint_path(model_name, modality)

    # Selecciona el dispositivo de inferencia.
    device = get_inference_device()

    # Construye la arquitectura correspondiente sin pesos ImageNet.
    model = build_model(model_name, pretrained=False).to(device)

    # Carga el checkpoint guardado durante entrenamiento.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Mantiene compatibilidad con versiones anteriores de PyTorch.
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extrae el state_dict si el checkpoint contiene metadatos.
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint

    # Valida que el checkpoint contenga pesos del modelo.
    if state_dict is None:
        raise RuntimeError(f"Checkpoint inválido: no contiene state_dict: {checkpoint_path}")

    # Carga los pesos en la arquitectura construida.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Detiene la carga si existen incompatibilidades de arquitectura o pesos.
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint incompatible para {model_name}. "
            f"missing_keys={list(missing)} unexpected_keys={list(unexpected)}"
        )

    # Coloca el modelo en modo evaluación para inferencia.
    model.eval()

    # Recupera el threshold calibrado o usa el configurado por defecto.
    threshold = float(checkpoint.get("threshold", get_settings().liveness_threshold)) if isinstance(checkpoint, dict) else get_settings().liveness_threshold

    # Recupera la época del checkpoint si existe.
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None

    # Recupera métricas de validación guardadas en el checkpoint si existen.
    val_metrics = checkpoint.get("val_metrics", {}) if isinstance(checkpoint, dict) else {}

    # Devuelve el modelo cargado junto con sus metadatos.
    return LoadedAntispoofModel(
        model_name=model_name,
        modality=modality,
        threshold=threshold,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=int(checkpoint_epoch) if checkpoint_epoch is not None else None,
        val_metrics=val_metrics,
        model=model,
        device=device,
    )


# Limpia la caché de modelos cargados.
def clear_model_cache() -> None:
    # Fuerza que los modelos se vuelvan a cargar en próximas solicitudes.
    load_antispoof_model.cache_clear()