import numpy as np


# Utilidades para preparar arreglos NumPy antes de convertirlos a tensores.
def normalize_imagenet(arr: np.ndarray) -> np.ndarray:
    # Escala los píxeles desde rango 0-255 hacia rango 0-1.
    arr = arr.astype("float32") / 255.0

    # Define las medias RGB estándar usadas por modelos preentrenados en ImageNet.
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")

    # Define las desviaciones estándar RGB usadas por modelos preentrenados en ImageNet.
    std = np.array([0.229, 0.224, 0.225], dtype="float32")

    # Normaliza cada canal RGB usando media y desviación estándar de ImageNet.
    return (arr - mean) / std