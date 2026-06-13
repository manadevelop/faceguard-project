from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
from ..exceptions import InvalidImageError


# Utilidades para leer, convertir y redimensionar imágenes usadas por el backend.
def read_image_bytes(data: bytes) -> Image.Image:
    # Intenta abrir los bytes recibidos como imagen válida.
    try:
        img = Image.open(BytesIO(data))

        # Corrige orientación EXIF y convierte la imagen a RGB.
        img = ImageOps.exif_transpose(img).convert("RGB")

        # Devuelve la imagen lista para preprocesamiento o inferencia.
        return img

    # Lanza un error controlado si el archivo no puede interpretarse como imagen.
    except Exception as exc:
        raise InvalidImageError("Archivo de imagen inválido") from exc


# Convierte una imagen PIL a arreglo NumPy en formato RGB.
def pil_to_numpy(img: Image.Image) -> np.ndarray:
    # Asegura tres canales RGB antes de convertir a NumPy.
    return np.asarray(img.convert("RGB"))


# Redimensiona una imagen PIL al tamaño cuadrado esperado por el modelo.
def resize_image(img: Image.Image, size: int = 224) -> Image.Image:
    # Aplica interpolación bilineal para mantener una escala suave.
    return img.resize((size, size), Image.Resampling.BILINEAR)