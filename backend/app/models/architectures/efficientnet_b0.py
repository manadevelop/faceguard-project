import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# Adaptación de EfficientNet-B0 para detección binaria LIVE/SPOOF.
class EfficientNetB0Binary(nn.Module):
    """EfficientNet-B0 con salida binaria compatible con checkpoints de entrenamiento."""

    def __init__(self, pretrained: bool = False):
        # Inicializa la clase base de PyTorch.
        super().__init__()

        # Selecciona pesos ImageNet si se habilita transferencia de aprendizaje.
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

        # Carga la arquitectura EfficientNet-B0 base.
        self.model = efficientnet_b0(weights=weights)

        # Obtiene el número de entradas de la capa clasificadora original.
        in_features = self.model.classifier[1].in_features

        # Reemplaza la salida multiclase por una salida binaria de un solo logit.
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        # Ejecuta el modelo y ajusta la salida para BCEWithLogitsLoss o inferencia binaria.
        return self.model(x).squeeze(1)