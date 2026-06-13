import cv2
import numpy as np
from PIL import Image


# Servicio para evaluar calidad básica de la imagen facial recibida.
class FaceQualityService:
    def assess(self, img: Image.Image) -> dict:
        # Convierte la imagen a escala de grises para analizar nitidez e iluminación.
        arr = np.asarray(img.convert("L"))

        # Calcula una métrica de nitidez usando la varianza del Laplaciano.
        blur = float(cv2.Laplacian(arr, cv2.CV_64F).var())

        # Calcula el brillo promedio de la imagen.
        brightness = float(np.mean(arr))

        # Devuelve indicadores de calidad útiles antes de la inferencia.
        return {
            "blur_score": blur,
            "brightness": brightness,
            "is_blurry": blur < 30,
            "is_dark": brightness < 35,
            "is_overexposed": brightness > 235,
            "is_valid": blur >= 20 and 20 <= brightness <= 245,
        }