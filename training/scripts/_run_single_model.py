#!/usr/bin/env python3
"""
FaceGuard - Ejecutor común para entrenar un único modelo.

Este script evita duplicar la lógica de argumentos en los wrappers
12_train_cnn_baseline.py, 13_train_efficientnet_b0.py,
14_train_mobilenetv3_small.py y 15_train_cdcn.py. Cada wrapper solo
indica el nombre del modelo y este archivo construye el comando final
hacia 08_train_all_models.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Raíz del proyecto y script central de entrenamiento.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / 'training' / 'scripts' / '08_train_all_models.py'


def run_single_model(model_name: str) -> None:
    """
    Ejecuta 08_train_all_models.py para un solo modelo.

    Parameters
    ----------
    model_name:
        Nombre técnico del modelo: cnn_baseline, efficientnet_b0,
        mobilenetv3_small o cdcn.
    """
    parser = argparse.ArgumentParser()

    # Modalidad de imagen: RGB (canal visible) o depth (mapa de profundidad)
    parser.add_argument('--modality', default='rgb', choices=['rgb', 'depth', 'all'])

    # Hiperparámetros de entrenamiento
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=2)

    # Semilla para reproducibilidad (afecta shuffling, dropout, inicialización)
    parser.add_argument('--seed', type=int, default=42)

    # Early stopping: detener si ACER de validación no mejora en N épocas
    parser.add_argument('--patience', type=int, default=5)

    # Flags opcionales para desactivar transferencia de aprendizaje, sampler o AMP.
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--no-weighted-sampler', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()

    # Comando final: se usa el mismo script central, pero restringido a un modelo.
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        '--models', model_name,
        '--modality', args.modality,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--image-size', str(args.image_size),
        '--lr', str(args.lr),
        '--weight-decay', str(args.weight_decay),
        '--num-workers', str(args.num_workers),
        '--seed', str(args.seed),
        '--patience', str(args.patience),
    ]

    if args.no_pretrained:
        cmd.append('--no-pretrained')
    if args.no_weighted_sampler:
        cmd.append('--no-weighted-sampler')
    if args.no_amp:
        cmd.append('--no-amp')

    print('=' * 90)
    print(f'Entrenamiento individual: {model_name}')
    print(' '.join(cmd))
    print('=' * 90)

    # check=True detiene el proceso si el entrenamiento falla.
    subprocess.run(cmd, check=True)


__all__ = ['run_single_model']
