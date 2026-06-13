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


# Servicio principal para ejecutar inferencia anti-spoofing con modelos entrenados.
class AntispoofService:
    """Inferencia anti-spoofing real usando checkpoints `.pt` entrenados.

    El backend web recibe crops RGB 224x224 desde la cámara. Por ese motivo,
    esta capa de servicio carga únicamente checkpoints RGB. Los checkpoints
    depth quedan para análisis experimental y reporte, no para la demo web.
    """

    def __init__(self):
        # Inicializa el servicio de preprocesamiento usado antes de la inferencia.
        self.preprocessing = PreprocessingService()

    def predict_image(self, img: Image.Image, model_name: str | None = None) -> dict[str, Any]:
        # Obtiene la configuración global de la aplicación.
        settings = get_settings()

        # Usa el modelo solicitado o el modelo por defecto configurado.
        model = (model_name or settings.liveness_default_model).lower().strip()

        # Valida que el modelo solicitado exista dentro de los modelos soportados.
        if model not in MODEL_NAMES:
            raise UnsupportedModelError(f"Modelo no soportado: {model}")

        # Ejecuta la inferencia real usando PyTorch y el checkpoint correspondiente.
        return self._predict_with_torch(img, model)

    def predict_frames(self, imgs: list[Image.Image], model_name: str | None = None) -> dict[str, Any]:
        # Valida que se haya recibido al menos un frame.
        if not imgs:
            raise ValueError("No se recibieron frames")

        # Ejecuta la predicción individual para cada frame recibido.
        results = [self.predict_image(img, model_name) for img in imgs]

        # Extrae los scores LIVE de todos los frames.
        scores = np.array([r["score"] for r in results], dtype="float32")

        # Usa el threshold del último resultado como referencia de decisión.
        threshold = float(results[-1]["threshold"])

        # Calcula el promedio de scores para tomar una decisión temporal.
        mean_score = float(scores.mean())

        # Decide si la secuencia completa se considera LIVE.
        is_live = mean_score >= threshold

        # Devuelve el resumen agregado de la verificación por frames.
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
        # Lista el estado de disponibilidad de los modelos anti-spoofing.
        rows = []

        # Recorre cada modelo registrado en el proyecto.
        for model_name in sorted(MODEL_NAMES):
            # Construye la respuesta base del modelo.
            row: dict[str, Any] = {
                "model": model_name,
                "modality": "rgb",
                "available": False,
                "checkpoint": None,
                "recommended_for_web": model_name == "efficientnet_b0",
            }

            # Intenta resolver el checkpoint RGB del modelo.
            try:
                checkpoint_path = resolve_checkpoint_path(model_name, "rgb")
                row["available"] = True
                row["checkpoint"] = str(checkpoint_path)

            # Registra el error si el checkpoint no existe.
            except FileNotFoundError as exc:
                row["error"] = str(exc)

            # Agrega el estado del modelo a la lista final.
            rows.append(row)

        # Devuelve todos los modelos con su disponibilidad.
        return rows

    def _predict_with_torch(self, img: Image.Image, model_name: str) -> dict[str, Any]:
        # Importa torch dentro del método para cargarlo solo cuando se requiere inferencia.
        import torch

        # Carga el modelo anti-spoofing y su checkpoint asociado.
        loaded = load_antispoof_model(model_name, "rgb")

        # Preprocesa la imagen al formato esperado por el modelo.
        arr = self.preprocessing.preprocess_for_model(img)

        # Convierte la imagen de NHWC a NCHW y la mueve al dispositivo de inferencia.
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(loaded.device)

        # Inicia la medición de latencia de inferencia.
        start = time.perf_counter()

        # Ejecuta inferencia sin cálculo de gradientes.
        with torch.no_grad():
            logits = loaded.model(tensor)
            score = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[0]

        # Calcula la latencia total en milisegundos.
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Convierte el score a float estándar de Python.
        score = float(score)

        # Obtiene el threshold calibrado desde el checkpoint.
        threshold = float(loaded.threshold)

        # Decide si la muestra supera el umbral de vida.
        is_live = score >= threshold

        # Devuelve el resultado completo de la inferencia anti-spoofing.
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