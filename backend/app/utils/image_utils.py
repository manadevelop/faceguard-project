from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
from ..exceptions import InvalidImageError

def read_image_bytes(data: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except Exception as exc:
        raise InvalidImageError("Archivo de imagen inválido") from exc

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))

def resize_image(img: Image.Image, size: int = 224) -> Image.Image:
    return img.resize((size, size), Image.Resampling.BILINEAR)
