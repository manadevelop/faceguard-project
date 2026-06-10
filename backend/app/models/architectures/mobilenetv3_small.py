import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3SmallBinary(nn.Module):
    """MobileNetV3-Small con salida binaria compatible con checkpoints de entrenamiento."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v3_small(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)
