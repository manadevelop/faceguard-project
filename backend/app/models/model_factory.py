from torch import nn


# Define los nombres de modelos que el backend puede construir para inferencia.
SUPPORTED_MODEL_NAMES = {"cnn_baseline", "efficientnet_b0", "mobilenetv3_small", "cdcn"}


# Construye dinámicamente la arquitectura solicitada para cargar checkpoints o ejecutar inferencia.
def build_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Construye la misma arquitectura usada durante entrenamiento.

    Las importaciones son perezosas para que el backend arranque sin cargar
    pesos hasta que se reciba una solicitud de inferencia.
    """
    # Normaliza el nombre recibido para evitar errores por mayúsculas o espacios.
    normalized = model_name.lower().strip()

    # Construye el modelo CNN Baseline.
    if normalized == "cnn_baseline":
        from .architectures.cnn_baseline import CNNBaseline
        return CNNBaseline()

    # Construye el modelo EfficientNet-B0 binario.
    if normalized == "efficientnet_b0":
        from .architectures.efficientnet_b0 import EfficientNetB0Binary
        return EfficientNetB0Binary(pretrained=pretrained)

    # Construye el modelo MobileNetV3-Small binario.
    if normalized == "mobilenetv3_small":
        from .architectures.mobilenetv3_small import MobileNetV3SmallBinary
        return MobileNetV3SmallBinary(pretrained=pretrained)

    # Construye el modelo CDCN con theta fijo usado durante el proyecto.
    if normalized == "cdcn":
        from .architectures.cdcn import CDCN
        return CDCN(theta=0.7)

    # Detiene la ejecución si se solicita un modelo no registrado.
    raise ValueError(f"Modelo no soportado: {model_name}")


# Genera los nombres de archivo esperados para ubicar checkpoints del modelo.
def get_expected_weight_names(model_name: str, modality: str = "rgb") -> list[str]:
    # Normaliza el nombre del modelo.
    model_name = model_name.lower().strip()

    # Normaliza la modalidad de entrada, por ejemplo rgb o depth.
    modality = modality.lower().strip()

    # Retorna posibles rutas compatibles con diferentes formas de guardar checkpoints.
    return [
        f"{model_name}_{modality}_best.pt",
        f"{model_name}_best.pt",
        f"{model_name}/{modality}_best.pt",
    ]