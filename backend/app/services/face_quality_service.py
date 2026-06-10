import cv2
import numpy as np
from PIL import Image

class FaceQualityService:
    def assess(self, img: Image.Image) -> dict:
        arr = np.asarray(img.convert("L"))
        blur = float(cv2.Laplacian(arr, cv2.CV_64F).var())
        brightness = float(np.mean(arr))
        return {
            "blur_score": blur,
            "brightness": brightness,
            "is_blurry": blur < 30,
            "is_dark": brightness < 35,
            "is_overexposed": brightness > 235,
            "is_valid": blur >= 20 and 20 <= brightness <= 245,
        }
