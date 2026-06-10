from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image

from ..config import get_settings
from ..constants import LABEL_REAL, LABEL_SPOOF, MODEL_NAMES
from ..exceptions import UnsupportedModelError
from .checkpoint_loader import load_antispoof_model, resolve_checkpoint_path
from .preprocessing_service import PreprocessingService


class AntispoofService:
    """Inferencia anti-spoofing real usando checkpoints `.pt` entrenados.

    El backend web recibe crops RGB 224x224 desde la cámara. Por ese motivo,
    esta capa de servicio carga únicamente checkpoints RGB. Los checkpoints
    depth quedan para análisis experimental y reporte, no para la demo web.
    """

    def __init__(self):
        self.preprocessing = PreprocessingService()

    def predict_image(self, img: Image.Image, model_name: str | None = None) -> dict[str, Any]:
        settings = get_settings()
        model = (model_name or settings.liveness_default_model).lower().strip()
        if model not in MODEL_NAMES:
            raise UnsupportedModelError(f"Modelo no soportado: {model}")
        return self._predict_with_torch(img, model)

    def predict_frames(self, imgs: list[Image.Image], model_name: str | None = None) -> dict[str, Any]:
        if not imgs:
            raise ValueError("No se recibieron frames")

        results = [self.predict_image(img, model_name) for img in imgs]
        scores = np.array([r["score"] for r in results], dtype="float32")
        threshold = float(results[-1]["threshold"])
        mean_score = float(scores.mean())
        is_live = mean_score >= threshold

        return {
            "is_live": is_live,
            "label": LABEL_REAL if is_live else LABEL_SPOOF,
            "score": round(mean_score, 6),
            "mean_score": round(mean_score, 6),
            "min_score": round(float(scores.min()), 6),
            "max_score": round(float(scores.max()), 6),
            "threshold": threshold,
            "model": results[-1]["model"],
            "modality": results[-1].get("modality", "rgb"),
            "num_frames": len(imgs),
            "checkpoint": results[-1].get("checkpoint"),
            "device": results[-1].get("device"),
        }

    def list_models(self) -> list[dict[str, Any]]:
        rows = []
        for model_name in sorted(MODEL_NAMES):
            row: dict[str, Any] = {
                "model": model_name,
                "modality": "rgb",
                "available": False,
                "checkpoint": None,
                "recommended_for_web": model_name == "efficientnet_b0",
            }
            try:
                checkpoint_path = resolve_checkpoint_path(model_name, "rgb")
                row["available"] = True
                row["checkpoint"] = str(checkpoint_path)
            except FileNotFoundError as exc:
                row["error"] = str(exc)
            rows.append(row)
        return rows

    def _predict_with_torch(self, img: Image.Image, model_name: str) -> dict[str, Any]:
        import torch

        loaded = load_antispoof_model(model_name, "rgb")
        arr = self.preprocessing.preprocess_for_model(img)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(loaded.device)

        start = time.perf_counter()
        with torch.no_grad():
            logits = loaded.model(tensor)
            score = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0

        score = float(score)
        threshold = float(loaded.threshold)
        is_live = score >= threshold

        return {
            "is_live": is_live,
            "label": LABEL_REAL if is_live else LABEL_SPOOF,
            "score": round(score, 6),
            "threshold": threshold,
            "model": loaded.model_name,
            "modality": loaded.modality,
            "checkpoint": str(loaded.checkpoint_path),
            "checkpoint_epoch": loaded.checkpoint_epoch,
            "device": str(loaded.device),
            "latency_ms": round(float(latency_ms), 4),
        }
