#!/usr/bin/env python3
"""
FaceGuard - Entrenamiento individual de EfficientNet-B0.

Este modelo usa transferencia de aprendizaje con pesos ImageNet y adapta
la cabeza final a clasificación binaria LIVE/SPOOF.
"""

from _run_single_model import run_single_model


# Punto de entrada del script cuando se ejecuta directamente desde consola.
if __name__ == '__main__':
    # Lanza el entrenamiento individual de EfficientNet-B0 usando la configuración común del pipeline.
    run_single_model('efficientnet_b0')