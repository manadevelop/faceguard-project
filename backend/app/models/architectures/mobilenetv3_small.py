import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# Adaptación de MobileNetV3-Small para clasificación binaria LIVE/SPOOF.
class MobileNetV3SmallBinary(nn.Module):
    """MobileNetV3-Small con salida binaria compatible con checkpoints de entrenamiento."""

    def __init__(self, pretrained: bool = False):
        # Inicializa la clase base de PyTorch.
        super().__init__()

        # Selecciona pesos ImageNet si se habilita transferencia de aprendizaje.
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None

        # Carga la arquitectura MobileNetV3-Small base.
        self.model = mobilenet_v3_small(weights=weights)

        # Obtiene el número de entradas de la última capa clasificadora.
        in_features = self.model.classifier[-1].in_features

        # Reemplaza la salida original por una salida binaria de un solo logit.
        self.model.classifier[-1] = nn.Linear(in_features, 1)

    def forward(self, x):
        # Ejecuta el modelo y ajusta la salida para clasificación binaria.
        return self.model(x).squeeze(1)