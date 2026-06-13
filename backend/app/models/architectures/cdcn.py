import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolución de diferencia central usada para resaltar patrones locales anti-spoofing.
class CentralDifferenceConv2d(nn.Module):
    """Central Difference Convolution usada por CDCN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        theta: float = 0.7,
    ):
        # Inicializa la clase base de PyTorch.
        super().__init__()

        # Controla la influencia del término de diferencia central.
        self.theta = theta

        # Define la convolución normal que se combinará con la diferencia central.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        # Calcula la salida de la convolución estándar.
        out_normal = self.conv(x)

        # Si theta es casi cero, se usa solo la convolución normal.
        if abs(self.theta) < 1e-8:
            return out_normal

        # Obtiene los pesos aprendidos de la convolución.
        weight = self.conv.weight

        # Suma el kernel espacial para construir el término de diferencia central.
        kernel_diff = weight.sum(dim=(2, 3), keepdim=True)

        # Aplica la convolución diferencial sobre la entrada.
        out_diff = F.conv2d(
            x,
            kernel_diff,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=0,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

        # Ajusta el tamaño espacial si la salida diferencial no coincide con la normal.
        if out_diff.shape[-2:] != out_normal.shape[-2:]:
            out_diff = F.interpolate(
                out_diff,
                size=out_normal.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Combina la convolución normal con la diferencia central ponderada.
        return out_normal - self.theta * out_diff


# Bloque CDC compuesto por dos convoluciones de diferencia central y una conexión residual.
class CDCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, theta: float = 0.7):
        # Inicializa la clase base de PyTorch.
        super().__init__()

        # Define el bloque principal con CDC, BatchNorm y activación ReLU.
        self.block = nn.Sequential(
            CentralDifferenceConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CentralDifferenceConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Define una conexión residual vacía cuando no se requiere ajuste de canales.
        self.shortcut = nn.Sequential()

        # Ajusta la conexión residual si cambia el número de canales.
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        # Suma la rama principal con la residual y aplica ReLU.
        return F.relu(self.block(x) + self.shortcut(x), inplace=True)


# Red CDCN compacta para detección de vida facial anti-spoofing.
class CDCN(nn.Module):
    """CDCN compacto para inferencia anti-spoofing."""

    def __init__(self, theta: float = 0.7):
        # Inicializa la clase base de PyTorch.
        super().__init__()

        # Bloque inicial que transforma la imagen RGB en mapas de características.
        self.stem = nn.Sequential(
            CentralDifferenceConv2d(3, 32, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Primera etapa CDC con reducción espacial por max pooling.
        self.stage1 = nn.Sequential(CDCBlock(32, 64, theta=theta), nn.MaxPool2d(2))

        # Segunda etapa CDC con mayor número de canales.
        self.stage2 = nn.Sequential(CDCBlock(64, 128, theta=theta), nn.MaxPool2d(2))

        # Tercera etapa CDC para extraer patrones faciales más complejos.
        self.stage3 = nn.Sequential(CDCBlock(128, 256, theta=theta), nn.MaxPool2d(2))

        # Cuarta etapa CDC antes de la clasificación final.
        self.stage4 = nn.Sequential(CDCBlock(256, 384, theta=theta), nn.MaxPool2d(2))

        # Cabeza clasificadora que produce un único logit LIVE/SPOOF.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # Procesa la imagen por el bloque inicial.
        x = self.stem(x)

        # Extrae características progresivas en la primera etapa.
        x = self.stage1(x)

        # Extrae características progresivas en la segunda etapa.
        x = self.stage2(x)

        # Extrae características progresivas en la tercera etapa.
        x = self.stage3(x)

        # Extrae características progresivas en la cuarta etapa.
        x = self.stage4(x)

        # Clasifica la muestra y ajusta la dimensión de salida.
        return self.head(x).squeeze(1)