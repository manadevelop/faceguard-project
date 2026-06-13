#!/usr/bin/env python3
"""
FaceGuard - Entrenamiento de modelos anti-spoofing.

Modelos soportados:
1. cnn_baseline
2. efficientnet_b0
3. mobilenetv3_small
4. cdcn

Ejemplos:

Entrenar CNN baseline RGB:
    python training/scripts/08_train_all_models.py --models cnn_baseline --modality rgb --epochs 1 --batch-size 16

Entrenar todos RGB:
    python training/scripts/08_train_all_models.py --models all --modality rgb --epochs 12 --batch-size 32

Entrenar todos RGB + depth:
    python training/scripts/08_train_all_models.py --models all --modality all --epochs 12 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
)
from tqdm import tqdm


# ============================================================
# PATHS
# Rutas centrales del proyecto, datasets procesados, salidas y modelos soportados.
# ============================================================

# Ubica la raíz del proyecto a partir de la ruta del script actual.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Agrupa las rutas principales del módulo de entrenamiento y sus datos.
TRAINING_DIR = PROJECT_ROOT / "training"
DATA_DIR = TRAINING_DIR / "data"

# Define las carpetas de entrada para datasets RGB y depth ya procesados.
PROCESSED_RGB_DIR = DATA_DIR / "processed"
PROCESSED_DEPTH_DIR = DATA_DIR / "processed_depth"

# Define las carpetas donde se guardan checkpoints, logs, figuras y reportes.
OUTPUTS_DIR = TRAINING_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Lista oficial de arquitecturas que puede entrenar este script.
MODEL_NAMES = ["cnn_baseline", "efficientnet_b0", "mobilenetv3_small", "cdcn"]


# ============================================================
# CONFIG
# Configuración central del entrenamiento: modelo, modalidad, batch, LR, etc.
# ============================================================

# Dataclass usado para transportar de forma ordenada todos los hiperparámetros.
@dataclass
class TrainConfig:
    model_name: str            # 'cnn_baseline' | 'efficientnet_b0' | 'mobilenetv3_small' | 'cdcn'
    modality: str              # 'rgb' | 'depth' — determina qué carpeta processed/ se usa
    data_dir: str              # Ruta absoluta a la carpeta con subcarpetas train/val/test
    image_size: int = 224      # Resolución cuadrada de entrada en píxeles
    epochs: int = 12           # Épocas máximas; early stopping puede detenerse antes
    batch_size: int = 32       # Imágenes por mini-batch
    learning_rate: float = 1e-4    # LR inicial para AdamW
    weight_decay: float = 1e-5     # Regularización L2 desacoplada (AdamW)
    num_workers: int = 2       # Workers paralelos del DataLoader para cargar imágenes
    seed: int = 42             # Semilla global; garantiza reproducibilidad entre runs
    patience: int = 5          # Épocas sin mejora de ACER_val antes de early stopping
    pretrained: bool = True    # Solo aplica a EfficientNet/MobileNet; ignorado en CNN Baseline
    use_weighted_sampler: bool = True  # Balancea clases desiguales (live vs spoof) en cada batch
    use_amp: bool = True       # Mixed Precision (float16 forward + float32 backward) en CUDA


# Crea las carpetas de salida necesarias antes de guardar logs, figuras y checkpoints.
def ensure_dirs() -> None:
    dirs = [
        OUTPUTS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        FIGURES_DIR / "training_curves",
        FIGURES_DIR / "confusion_matrices",
        FIGURES_DIR / "roc_curves",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# Fija semillas para mejorar reproducibilidad entre ejecuciones.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Selecciona automáticamente el dispositivo disponible: CUDA, Apple MPS o CPU.
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# DATASET
# Dataset personalizado: lee carpetas train/val/test con subcarpetas live y spoof.
# ============================================================

# Clase Dataset de PyTorch para cargar imágenes faciales y etiquetas binarias.
class FaceGuardImageDataset(Dataset):
    """
    Dataset binario:
      live  -> 1
      spoof -> 0
    """

    # Constructor: recibe raíz del dataset, split y transformaciones.
    def __init__(self, root_dir: Path, split: str, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        split_dir = self.root_dir / split

        if not split_dir.exists():
            raise RuntimeError(f"No existe la carpeta del split: {split_dir}")

        # Lista interna de muestras: cada elemento será (ruta_imagen, etiqueta).
        self.samples: List[Tuple[Path, int]] = []

        # Estructura esperada: split/live para rostros reales y split/spoof para ataques.
        live_dir = split_dir / "live"
        spoof_dir = split_dir / "spoof"

        # Recorre recursivamente imágenes LIVE y asigna etiqueta 1.
        if live_dir.exists():
            for p in sorted(live_dir.rglob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((p, 1))

        # Recorre recursivamente imágenes SPOOF y asigna etiqueta 0.
        if spoof_dir.exists():
            for p in sorted(spoof_dir.rglob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((p, 0))

        if len(self.samples) == 0:
            raise RuntimeError(f"No se encontraron imágenes en {split_dir}")

    # Devuelve el número total de muestras disponibles.
    def __len__(self) -> int:
        return len(self.samples)

    # Carga una muestra individual: imagen transformada, etiqueta y ruta original.
    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        # Se fuerza RGB para que todas las arquitecturas reciban 3 canales.
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # BCEWithLogitsLoss espera etiquetas float para clasificación binaria.
        y = torch.tensor(label, dtype=torch.float32)

        return img, y, str(path)


# Define transformaciones: con augmentation para train y deterministas para val/test.
def get_transforms(image_size: int, train: bool) -> transforms.Compose:
    """
    Devuelve el pipeline de transformaciones según el modo.

    Entrenamiento: incluye augmentation para diversificar los datos y evitar overfitting.
    Validación/Test: solo redimensiona y normaliza (sin aleatoriedad, para evaluación consistente).
    """
    if train:
        return transforms.Compose(
            [
                # Redimensiona a 224×224; necesario porque las imágenes del dataset tienen tamaños variados
                transforms.Resize((image_size, image_size)),

                # Volteo horizontal aleatorio: una cara real de perfil izquierdo ≈ perfil derecho
                transforms.RandomHorizontalFlip(p=0.5),

                # Variaciones de color (55% de probabilidad): simula distintas iluminaciones y cámaras
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.20,   # ±20% de brillo
                            contrast=0.20,     # ±20% de contraste
                            saturation=0.15,   # ±15% de saturación
                            hue=0.03,          # ±3% de tono (sutil, evita colores irreales)
                        )
                    ],
                    p=0.55,
                ),

                # Desenfoque gaussiano leve (20% de probabilidad): simula cámaras fuera de foco
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=3,
                            sigma=(0.1, 1.5),  # sigma pequeño para no destruir texturas de piel
                        )
                    ],
                    p=0.20,
                ),

                # Rotación aleatoria de hasta ±8°: invariancia a inclinaciones leves de cabeza
                transforms.RandomRotation(degrees=8),

                # Convierte PIL Image a tensor float32 en rango [0, 1] con forma [C, H, W]
                transforms.ToTensor(),

                # Normalización con media y std de ImageNet; escala los valores a ~[-2, 2]
                # Se usa aunque la CNN Baseline no use pesos de ImageNet: estabiliza el entrenamiento
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    # En validación/test no se aplica augmentation: solo resize + normalización.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# Construye DataLoaders y aplica WeightedRandomSampler para manejar desbalance LIVE/SPOOF.
def build_dataloaders(cfg: TrainConfig):
    data_dir = Path(cfg.data_dir)

    train_ds = FaceGuardImageDataset(
        root_dir=data_dir,
        split="train",
        transform=get_transforms(cfg.image_size, train=True),
    )

    val_ds = FaceGuardImageDataset(
        root_dir=data_dir,
        split="val",
        transform=get_transforms(cfg.image_size, train=False),
    )

    test_ds = FaceGuardImageDataset(
        root_dir=data_dir,
        split="test",
        transform=get_transforms(cfg.image_size, train=False),
    )

    # Se extraen etiquetas del train set para contar clases y construir pesos.
    train_labels = [label for _, label in train_ds.samples]

    # Cuenta cuántas imágenes hay por clase: {0: n_spoof, 1: n_live}
    class_counts = {
        0: int(sum(1 for y in train_labels if y == 0)),
        1: int(sum(1 for y in train_labels if y == 1)),
    }

    sampler = None
    shuffle = True  # shuffle normal si no se usa WeightedRandomSampler

    if cfg.use_weighted_sampler:
        # Peso por clase = 1 / cantidad; las clases pequeñas reciben peso mayor
        # Ejemplo: 1000 live → peso 0.001 | 300 spoof → peso 0.0033
        # Esto hace que el sampler elija spoof con más frecuencia para balancear los batches
        weights_per_class = {
            cls: 1.0 / max(count, 1)
            for cls, count in class_counts.items()
        }

        # Asigna el peso de su clase a cada muestra individual
        sample_weights = [weights_per_class[y] for y in train_labels]

        # WeightedRandomSampler: muestrea con reemplazo según los pesos calculados
        # Resultado: cada batch tiene aproximadamente 50% live / 50% spoof sin importar el desbalance real
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,  # permite repetir imágenes para compensar la clase minoritaria
        )

        # shuffle=False porque el sampler ya se encarga del orden aleatorio balanceado
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_counts


# ============================================================
# MODELS
# Definición de las arquitecturas evaluadas en FaceGuard.
# ============================================================

# CNN propia usada como línea base entrenada desde cero.
class CNNBaseline(nn.Module):
    """
    Red convolucional construida desde cero para clasificación binaria REAL/SPOOF.

    Arquitectura: 5 bloques convolucionales (extracción de características)
                  + 1 clasificador fully-connected (decisión final).

    Flujo de dimensiones con entrada [B, 3, 224, 224]:
        Bloque 1 → [B, 32,  112, 112]
        Bloque 2 → [B, 64,   56,  56]
        Bloque 3 → [B, 128,  28,  28]
        Bloque 4 → [B, 256,  14,  14]
        Bloque 5 → [B, 384,   7,   7]
        AvgPool  → [B, 384,   1,   1] → Flatten → [B, 384]
        Linear   → [B, 128] → Linear → [B, 1]  → squeeze → [B]

    No usa pesos preentrenados: aprende todo desde cero con los datos de FaceGuard.
    """

    def __init__(self):
        super().__init__()

        # ── Parte extractora de características ──────────────────────────────
        # Cada bloque: Conv → BatchNorm → ReLU → MaxPool(2)
        # Los canales se duplican aprox. en cada bloque mientras la resolución se reduce a la mitad.
        # bias=False porque BatchNorm ya aprende su propio sesgo (parámetro β).
        self.features = nn.Sequential(
            # Bloque 1: detecta bordes y gradientes de bajo nivel
            # padding=1 mantiene resolución espacial antes del MaxPool
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),   # normaliza activaciones dentro del batch → entrenamiento más estable
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 224 → 112

            # Bloque 2: detecta texturas simples (puntos, líneas cortas)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 112 → 56

            # Bloque 3: detecta patrones más complejos (texturas de piel, tramas de papel/pantalla)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 56 → 28

            # Bloque 4: detecta características de alto nivel (regiones faciales, artefactos de impresión)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 28 → 14

            # Bloque 5: integra contexto global (¿tiene la imagen los marcadores de un ataque?)
            # Sin MaxPool al final: el AdaptiveAvgPool del clasificador maneja la reducción final
            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # Resolución aquí: 14 → 7 (sin pool explícito; AdaptiveAvgPool lo reducirá a 1×1)
        )

        # ── Parte clasificadora ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            # Reduce cualquier resolución espacial a 1×1: convierte [B, 384, 7, 7] → [B, 384, 1, 1]
            nn.AdaptiveAvgPool2d(1),

            # Elimina las dimensiones espaciales: [B, 384, 1, 1] → [B, 384]
            nn.Flatten(),

            # Dropout(0.35): apaga 35% de las neuronas al azar durante entrenamiento
            # Fuerza a la red a no depender de ningún descriptor específico → reduce overfitting
            nn.Dropout(0.35),

            # Proyecta el vector de 384 dims a 128; aprende combinaciones de características
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),

            # Dropout más leve cerca de la salida para no perder demasiada capacidad
            nn.Dropout(0.25),

            # Capa de salida: produce 1 logit escalar (sin Sigmoid aquí)
            # La Sigmoid se aplica implícitamente dentro de BCEWithLogitsLoss durante el entrenamiento
            # y explícitamente (sigmoid()) durante la inferencia para obtener probabilidad
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # Extrae características jerárquicas a través de los 5 bloques convolucionales
        x = self.features(x)
        # Clasifica: pooling → flatten → dropout → linear → logit
        x = self.classifier(x)
        # squeeze(1): elimina la dimensión de clase → [B, 1] → [B]
        # BCEWithLogitsLoss espera un tensor 1D de logits, no 2D
        return x.squeeze(1)


# Convolución de diferencia central: resalta cambios locales útiles para anti-spoofing.
class CentralDifferenceConv2d(nn.Module):
    """
    Central Difference Convolution simplificada para CDCN.
    """

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
        super().__init__()

        self.theta = theta

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
        out_normal = self.conv(x)

        if abs(self.theta) < 1e-8:
            return out_normal

        weight = self.conv.weight
        kernel_diff = weight.sum(dim=(2, 3), keepdim=True)

        out_diff = F.conv2d(
            x,
            kernel_diff,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=0,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

        if out_diff.shape[-2:] != out_normal.shape[-2:]:
            out_diff = F.interpolate(
                out_diff,
                size=out_normal.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return out_normal - self.theta * out_diff


# Bloque residual basado en convoluciones de diferencia central.
class CDCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, theta: float = 0.7):
        super().__init__()

        self.block = nn.Sequential(
            CentralDifferenceConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                theta=theta,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            CentralDifferenceConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                theta=theta,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x), inplace=True)


# Arquitectura CDCN compacta especializada para detección de ataques de presentación.
class CDCN(nn.Module):
    """
    CDCN compacto para face anti-spoofing.
    """

    def __init__(self, theta: float = 0.7):
        super().__init__()

        self.stem = nn.Sequential(
            CentralDifferenceConv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
                theta=theta,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            CDCBlock(32, 64, theta=theta),
            nn.MaxPool2d(2),
        )

        self.stage2 = nn.Sequential(
            CDCBlock(64, 128, theta=theta),
            nn.MaxPool2d(2),
        )

        self.stage3 = nn.Sequential(
            CDCBlock(128, 256, theta=theta),
            nn.MaxPool2d(2),
        )

        self.stage4 = nn.Sequential(
            CDCBlock(256, 384, theta=theta),
            nn.MaxPool2d(2),
        )

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
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x).squeeze(1)


# Fábrica de modelos: reconstruye la arquitectura indicada por nombre.
def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower().strip()

    if model_name == "cnn_baseline":
        return CNNBaseline()

    if model_name == "efficientnet_b0":
        # Transfer learning: pesos ImageNet si pretrained=True.
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b0(weights=weights)
        in_features = base_model.classifier[1].in_features
        # Se reemplaza la cabeza ImageNet de 1000 clases por una salida binaria LIVE/SPOOF.
        base_model.classifier[1] = nn.Linear(in_features, 1)

        class EfficientNetB0Binary(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model(x).squeeze(1)

        return EfficientNetB0Binary(base_model)

    # Construcción de MobileNetV3-Small adaptado a salida binaria.
    if model_name == "mobilenetv3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v3_small(weights=weights)
        in_features = base_model.classifier[-1].in_features
        base_model.classifier[-1] = nn.Linear(in_features, 1)

        class MobileNetV3SmallBinary(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model(x).squeeze(1)

        return MobileNetV3SmallBinary(base_model)

    if model_name == "cdcn":
        return CDCN(theta=0.7)

    raise ValueError(f"Modelo no soportado: {model_name}")

# ============================================================
# METRICS
# ============================================================

# Convierte logits a probabilidades en numpy.
def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


# Calcula métricas clásicas y biométricas para un threshold dado.
def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / max(tn + fp, 1)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    # APCER: ataques aceptados como live = FP / total spoof
    # BPCER: usuarios reales rechazados = FN / total live
    # ── Métricas biométricas estándar ISO/IEC 30107-3 ────────────────────────
    # APCER (Attack Presentation Classification Error Rate):
    #   Fracción de ataques (spoof) que el modelo clasificó como cara real.
    #   FP = spoofs aceptados. FP+TN = total de spoofs reales.
    #   APCER bajo → el sistema rechaza bien los ataques.
    apcer = fp / max(fp + tn, 1)

    # BPCER (Bona Fide Presentation Classification Error Rate):
    #   Fracción de caras reales que el modelo rechazó como spoof.
    #   FN = reales rechazados. FN+TP = total de caras reales.
    #   BPCER bajo → el sistema no bloquea a usuarios legítimos.
    bpcer = fn / max(fn + tp, 1)

    # ACER (Average Classification Error Rate):
    #   Promedio de los dos errores. Es la métrica principal del proyecto.
    #   Se minimiza durante la búsqueda del umbral óptimo.
    #   ACER = 0 → sistema perfecto. ACER = 0.5 → equivalente a adivinar al azar.
    acer = (apcer + bpcer) / 2.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# Busca el threshold que minimiza ACER en validación.
def find_best_threshold_by_acer(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.05, 0.95, 181)

    best_t = 0.5
    best_metrics = calculate_metrics(y_true, y_prob, threshold=0.5)

    for t in thresholds:
        m = calculate_metrics(y_true, y_prob, threshold=float(t))

        if m["acer"] < best_metrics["acer"]:
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics


# ============================================================
# TRAIN / EVAL
# ============================================================

# Ejecuta inferencia sobre un DataLoader y mide latencia promedio por muestra.
def run_inference(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []
    all_paths = []

    start = time.time()

    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device)
            logits = model(x)

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.numpy())
            all_paths.extend(paths)

    elapsed = time.time() - start

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0).astype(int)
    probs_np = sigmoid_np(logits_np)

    latency_ms = (elapsed / max(len(labels_np), 1)) * 1000.0

    return labels_np, probs_np, all_paths, latency_ms


# Ejecuta una época de entrenamiento con forward, loss, backward y optimizer.step().
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    use_amp: bool = True,
) -> float:
    """
    Ejecuta una época completa de entrenamiento y devuelve la pérdida promedio.

    Soporta Automatic Mixed Precision (AMP) en CUDA:
    - forward + loss: float16 (más rápido, menos VRAM)
    - backward (gradientes): float32 (más estable numéricamente)
    El GradScaler escala la pérdida para evitar underflow de gradientes en float16.
    """
    model.train()  # Activa BatchNorm en modo entrenamiento y habilita Dropout

    losses = []

    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        # set_to_none=True libera memoria de los gradientes anteriores (más eficiente que zero_grad)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and use_amp and device.type == "cuda":
            # ── Rama AMP (GPU) ────────────────────────────────────────────────
            with torch.cuda.amp.autocast():
                logits = model(x)          # forward en float16
                loss = criterion(logits, y) # BCEWithLogitsLoss en float16

            scaler.scale(loss).backward()  # escala la pérdida antes del backward para evitar underflow
            scaler.step(optimizer)         # desescala los gradientes y actualiza pesos
            scaler.update()                # ajusta el factor de escala para la siguiente iteración
        else:
            # ── Rama estándar (CPU / MPS) ─────────────────────────────────────
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()    # calcula gradientes ∂L/∂w para todos los parámetros
            optimizer.step()   # actualiza pesos: w = w - lr * gradiente_adaptado (AdamW)

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("nan")


# Calcula la pérdida promedio en validación/test sin actualizar pesos.
def evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()

    losses = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("nan")


# Genera curva combinada de loss, ACER y AUC por época.
def plot_training_curves(history: List[Dict], out_path: Path) -> None:
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acer = [h["val_acer"] for h in history]
    val_auc = [h["val_auc"] for h in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.plot(epochs, val_acer, label="val_acer")
    plt.plot(epochs, val_auc, label="val_auc")
    plt.xlabel("Epoch")
    plt.ylabel("Valor")
    plt.title("Curvas de entrenamiento")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


# Genera matriz de confusión usando el threshold seleccionado.
def plot_confusion(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_path: Path,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Matriz de confusión")
    plt.xticks([0, 1], ["SPOOF", "LIVE"])
    plt.yticks([0, 1], ["SPOOF", "LIVE"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


# Genera curva ROC y calcula AUC.
def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        return

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


# Entrena un modelo completo: datos, modelo, loss, optimizador, validación, checkpoint y test.
def train_model(cfg: TrainConfig) -> Dict:
    ensure_dirs()
    set_seed(cfg.seed)

    device = get_device()

    print("\n" + "=" * 90)
    print(f"Entrenando {cfg.model_name} | {cfg.modality} | device={device}")
    print("=" * 90)

    train_loader, val_loader, test_loader, class_counts = build_dataloaders(cfg)

    print("Class counts train:", class_counts)

    # Construye el modelo solicitado y lo mueve al dispositivo seleccionado.
    model = build_model(cfg.model_name, pretrained=cfg.pretrained).to(device)

    # ── Función de pérdida con peso de clase positiva ─────────────────────────
    num_spoof = class_counts.get(0, 1)
    num_live  = class_counts.get(1, 1)
    # pos_weight pondera más la pérdida de la clase positiva (live=1) cuando hay más spoof que live
    # Ejemplo: 1000 spoof, 300 live → pos_weight = 3.33 → la pérdida de un live mal clasificado
    # vale 3.33× más que la de un spoof mal clasificado
    pos_weight_value = num_spoof / max(num_live, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    # BCEWithLogitsLoss = Sigmoid + Binary Cross-Entropy, numéricamente más estable que separarlos
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizador AdamW ─────────────────────────────────────────────────────
    # AdamW = Adam con weight decay desacoplado: w = w - lr*(m̂/(√v̂+ε)) - lr*λ*w
    # Mejora la regularización respecto a Adam estándar (que acopla weight decay al gradiente)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,      # 1e-4 inicial
        weight_decay=cfg.weight_decay,  # λ=1e-5 (penaliza pesos grandes)
    )

    # ── Scheduler: reduce LR si el ACER de validación no mejora ──────────────
    # mode='min': monitorea una métrica que queremos minimizar (ACER)
    # patience=2: espera 2 épocas sin mejora antes de reducir
    # factor=0.5: lr_nuevo = lr_actual * 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        factor=0.5,
    )

    # GradScaler para AMP: escala la pérdida para evitar underflow de gradientes en float16
    # Solo se activa en GPU CUDA; en CPU/MPS no hay beneficio de AMP
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and cfg.use_amp else None

    model_dir = CHECKPOINTS_DIR / cfg.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    best_path = model_dir / f"{cfg.modality}_best.pt"
    last_path = model_dir / f"{cfg.modality}_last.pt"

    best_acer = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=cfg.use_amp,
        )

        val_loss = evaluate_loss(model, val_loader, criterion, device)

        y_val, p_val, _, val_latency_ms = run_inference(model, val_loader, device)
        threshold, val_metrics = find_best_threshold_by_acer(y_val, p_val)

        scheduler.step(val_metrics["acer"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_threshold": threshold,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1_score"],
            "val_auc": val_metrics["roc_auc"],
            "val_apcer": val_metrics["apcer"],
            "val_bpcer": val_metrics["bpcer"],
            "val_acer": val_metrics["acer"],
            "val_latency_ms": val_latency_ms,
            "lr": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f} | "
            f"val_acer={val_metrics['acer']:.4f} | "
            f"threshold={threshold:.3f}"
        )

        # Checkpoint completo: pesos, configuración, threshold y métricas de validación.
        checkpoint_payload = {
            "model_name": cfg.model_name,
            "modality": cfg.modality,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "threshold": threshold,
            "val_metrics": val_metrics,
            "class_counts": class_counts,
        }

        torch.save(checkpoint_payload, last_path)

        # Se guarda como best.pt únicamente si mejora ACER de validación.
        if val_metrics["acer"] < best_acer:
            best_acer = val_metrics["acer"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint_payload, best_path)
        else:
            epochs_without_improvement += 1

        # Early stopping: detiene si no hay mejora durante varias épocas.
        if epochs_without_improvement >= cfg.patience:
            print(f"Early stopping en epoch {epoch}. Mejor epoch={best_epoch}")
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    best_threshold = float(checkpoint["threshold"])

    # Evaluación final en test usando el mejor threshold guardado.
    y_test, p_test, test_paths, test_latency_ms = run_inference(model, test_loader, device)
    test_metrics = calculate_metrics(y_test, p_test, threshold=best_threshold)
    test_metrics["latency_ms_per_sample"] = float(test_latency_ms)

    history_df = pd.DataFrame(history)
    history_path = LOGS_DIR / f"{cfg.model_name}_{cfg.modality}_history.csv"
    history_df.to_csv(history_path, index=False)

    pred_df = pd.DataFrame(
        {
            "path": test_paths,
            "y_true": y_test,
            "y_prob_live": p_test,
            "y_pred": (p_test >= best_threshold).astype(int),
        }
    )

    pred_path = REPORTS_DIR / f"predictions_{cfg.model_name}_{cfg.modality}.csv"
    pred_df.to_csv(pred_path, index=False)

    plot_training_curves(
        history,
        FIGURES_DIR / "training_curves" / f"{cfg.model_name}_{cfg.modality}.png",
    )

    plot_confusion(
        y_test,
        p_test,
        best_threshold,
        FIGURES_DIR / "confusion_matrices" / f"{cfg.model_name}_{cfg.modality}.png",
    )

    plot_roc(
        y_test,
        p_test,
        FIGURES_DIR / "roc_curves" / f"{cfg.model_name}_{cfg.modality}.png",
    )

    result = {
        "model_name": cfg.model_name,
        "modality": cfg.modality,
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path.relative_to(PROJECT_ROOT)),
        "threshold": best_threshold,
        "class_counts_train": class_counts,
        "test_metrics": test_metrics,
        "history_path": str(history_path.relative_to(PROJECT_ROOT)),
        "predictions_path": str(pred_path.relative_to(PROJECT_ROOT)),
    }

    result_path = REPORTS_DIR / f"metrics_{cfg.model_name}_{cfg.modality}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nResultado test:")
    print(json.dumps(test_metrics, indent=2))

    return result


# ============================================================
# MAIN
# ============================================================

# Parsea nombres de modelos desde consola: all o lista separada por comas.
def parse_models(value: str) -> List[str]:
    value = value.strip().lower()

    if value == "all":
        return MODEL_NAMES

    models = [m.strip().lower() for m in value.split(",") if m.strip()]

    for m in models:
        if m not in MODEL_NAMES:
            raise ValueError(f"Modelo inválido: {m}. Modelos válidos: {MODEL_NAMES}")

    return models


# Parsea modalidad desde consola: rgb, depth o all.
def parse_modalities(value: str) -> List[str]:
    value = value.strip().lower()

    if value == "all":
        return ["rgb", "depth"]

    if value not in {"rgb", "depth"}:
        raise ValueError("modality debe ser: rgb, depth o all")

    return [value]


# Punto de entrada: lee argumentos y ejecuta entrenamiento por modelo y modalidad.
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", default="all", help="all o lista separada por coma")
    parser.add_argument("--modality", default="rgb", choices=["rgb", "depth", "all"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)

    models = parse_models(args.models)
    modalities = parse_modalities(args.modality)

    all_results = []

    for modality in modalities:
        if modality == "rgb":
            data_dir = PROCESSED_RGB_DIR
        else:
            data_dir = PROCESSED_DEPTH_DIR

        if not data_dir.exists():
            raise RuntimeError(f"No existe data_dir: {data_dir}")

        for model_name in models:
            cfg = TrainConfig(
                model_name=model_name,
                modality=modality,
                data_dir=str(data_dir),
                image_size=args.image_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                seed=args.seed,
                patience=args.patience,
                pretrained=not args.no_pretrained,
                use_weighted_sampler=not args.no_weighted_sampler,
                use_amp=not args.no_amp,
            )

            result = train_model(cfg)

            test_metrics = dict(result["test_metrics"])
            test_metrics.pop("threshold", None)

            row = {
                "model_name": result["model_name"],
                "modality": result["modality"],
                "best_epoch": result["best_epoch"],
                "selected_threshold": result["threshold"],
                **test_metrics,
                "best_checkpoint": result["best_checkpoint"],
            }

            all_results.append(row)

    comparison = pd.DataFrame(all_results)
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n" + "=" * 90)
    print("COMPARACIÓN FINAL")
    print("=" * 90)
    print(comparison.to_string(index=False))
    print(f"\nArchivo: {comparison_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()