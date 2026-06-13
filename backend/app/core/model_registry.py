# Registro central de modelos anti-spoofing disponibles en FaceGuard.
MODEL_REGISTRY = {
    # Configuración del modelo CNN Baseline entrenado desde cero.
    "cnn_baseline": {
        "display_name": "CNN baseline",
        "default_modality": "rgb",
        "recommended_for_web": False,
    },

    # Configuración del modelo EfficientNet-B0 seleccionado como modelo operativo web.
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "default_modality": "rgb",
        "recommended_for_web": True,
    },

    # Configuración del modelo liviano MobileNetV3-Small.
    "mobilenetv3_small": {
        "display_name": "MobileNetV3-Small",
        "default_modality": "rgb",
        "recommended_for_web": False,
    },

    # Configuración del modelo CDCN especializado en anti-spoofing facial.
    "cdcn": {
        "display_name": "CDCN",
        "default_modality": "rgb",
        "recommended_for_web": False,
    },
}