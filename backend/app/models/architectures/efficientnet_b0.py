import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0Binary(nn.Module):
    """EfficientNet-B0 con salida binaria compatible con checkpoints de entrenamiento."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)
