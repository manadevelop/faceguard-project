from __future__ import annotations

import numpy as np
from PIL import Image

from ..config import get_settings
from ..utils.image_utils import resize_image, pil_to_numpy
from ..utils.tensor_utils import normalize_imagenet


class PreprocessingService:
    def preprocess_for_model(self, img: Image.Image) -> np.ndarray:
        """Preprocesa una imagen PIL a formato NHWC normalizado ImageNet.

        Devuelve shape: (1, H, W, 3). El servicio de inferencia convierte a NCHW.
        """
        size = get_settings().expected_crop_size
        img = resize_image(img.convert("RGB"), size)
        arr = pil_to_numpy(img)
        arr = normalize_imagenet(arr)
        return arr[None, ...]
