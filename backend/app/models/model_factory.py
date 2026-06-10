from torch import nn

SUPPORTED_MODEL_NAMES = {"cnn_baseline", "efficientnet_b0", "mobilenetv3_small", "cdcn"}


def build_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Construye la misma arquitectura usada durante entrenamiento.

    Las importaciones son perezosas para que el backend arranque sin cargar
    pesos hasta que se reciba una solicitud de inferencia.
    """
    normalized = model_name.lower().strip()

    if normalized == "cnn_baseline":
        from .architectures.cnn_baseline import CNNBaseline
        return CNNBaseline()

    if normalized == "efficientnet_b0":
        from .architectures.efficientnet_b0 import EfficientNetB0Binary
        return EfficientNetB0Binary(pretrained=pretrained)

    if normalized == "mobilenetv3_small":
        from .architectures.mobilenetv3_small import MobileNetV3SmallBinary
        return MobileNetV3SmallBinary(pretrained=pretrained)

    if normalized == "cdcn":
        from .architectures.cdcn import CDCN
        return CDCN(theta=0.7)

    raise ValueError(f"Modelo no soportado: {model_name}")


def get_expected_weight_names(model_name: str, modality: str = "rgb") -> list[str]:
    model_name = model_name.lower().strip()
    modality = modality.lower().strip()
    return [
        f"{model_name}_{modality}_best.pt",
        f"{model_name}_best.pt",
        f"{model_name}/{modality}_best.pt",
    ]
