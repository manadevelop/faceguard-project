import torch.nn as nn


# Arquitectura CNN base usada como línea de referencia para clasificación LIVE/SPOOF.
class CNNBaseline(nn.Module):
    """CNN baseline usada en entrenamiento para clasificación LIVE/SPOOF."""

    def __init__(self):
        # Inicializa la clase base de PyTorch.
        super().__init__()

        # Bloque extractor de características basado en convoluciones, normalización y pooling.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        # Cabeza clasificadora que reduce las características y produce un único logit.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # Ejecuta extracción de características, clasificación binaria y ajusta la dimensión de salida.
        return self.classifier(self.features(x)).squeeze(1)