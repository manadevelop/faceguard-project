from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ..config import get_settings
from ..exceptions import UnsupportedModelError
from ..models.model_factory import build_model, get_expected_weight_names, SUPPORTED_MODEL_NAMES


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


def get_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint_path(model_name: str, modality: str = "rgb") -> Path:
    settings = get_settings()
    weights_dir = Path(settings.weights_dir)

    for candidate in get_expected_weight_names(model_name, modality):
        path = weights_dir / candidate
        if path.exists() and path.is_file():
            return path

    expected = ", ".join(str(weights_dir / name) for name in get_expected_weight_names(model_name, modality))
    raise FileNotFoundError(
        f"No se encontró checkpoint para model={model_name}, modality={modality}. "
        f"Ubica el .pt en una de estas rutas: {expected}"
    )


@lru_cache(maxsize=16)
def load_antispoof_model(model_name: str, modality: str = "rgb") -> LoadedAntispoofModel:
    model_name = model_name.lower().strip()
    modality = modality.lower().strip()

    if model_name not in SUPPORTED_MODEL_NAMES:
        raise UnsupportedModelError(f"Modelo no soportado: {model_name}")
    if modality != "rgb":
        raise UnsupportedModelError("El backend web solo acepta inferencia RGB desde webcam.")

    checkpoint_path = resolve_checkpoint_path(model_name, modality)
    device = get_inference_device()

    model = build_model(model_name, pretrained=False).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        raise RuntimeError(f"Checkpoint inválido: no contiene state_dict: {checkpoint_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint incompatible para {model_name}. "
            f"missing_keys={list(missing)} unexpected_keys={list(unexpected)}"
        )

    model.eval()

    threshold = float(checkpoint.get("threshold", get_settings().liveness_threshold)) if isinstance(checkpoint, dict) else get_settings().liveness_threshold
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    val_metrics = checkpoint.get("val_metrics", {}) if isinstance(checkpoint, dict) else {}

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


def clear_model_cache() -> None:
    load_antispoof_model.cache_clear()
