import numpy as np

def normalize_imagenet(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    return (arr - mean) / std
