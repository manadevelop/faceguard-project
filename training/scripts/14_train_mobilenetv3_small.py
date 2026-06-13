#!/usr/bin/env python3
"""
FaceGuard - Entrenamiento individual de MobileNetV3-Small.

Se usa como alternativa liviana para comparar precisión y latencia frente
a EfficientNet-B0 y CNN Baseline.
"""

from _run_single_model import run_single_model


# Punto de entrada del script cuando se ejecuta directamente desde consola.
if __name__ == '__main__':
    # Lanza el entrenamiento individual de MobileNetV3-Small usando la configuración común del pipeline.
    run_single_model('mobilenetv3_small')