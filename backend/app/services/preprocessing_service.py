from __future__ import annotations

import numpy as np
from PIL import Image

from ..config import get_settings
from ..utils.image_utils import resize_image, pil_to_numpy
from ..utils.tensor_utils import normalize_imagenet


# Servicio encargado de preparar imágenes antes de enviarlas al modelo.
class PreprocessingService:
    def preprocess_for_model(self, img: Image.Image) -> np.ndarray:
        """Preprocesa una imagen PIL a formato NHWC normalizado ImageNet.

        Devuelve shape: (1, H, W, 3). El servicio de inferencia convierte a NCHW.
        """
        # Obtiene el tamaño de crop esperado por el modelo.
        size = get_settings().expected_crop_size

        # Convierte la imagen a RGB y la redimensiona al tamaño requerido.
        img = resize_image(img.convert("RGB"), size)

        # Convierte la imagen PIL a arreglo NumPy.
        arr = pil_to_numpy(img)

        # Normaliza la imagen usando medias y desviaciones de ImageNet.
        arr = normalize_imagenet(arr)

        # Agrega la dimensión batch para inferencia.
        return arr[None, ...]