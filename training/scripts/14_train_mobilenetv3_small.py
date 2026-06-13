#!/usr/bin/env python3
"""
FaceGuard - Entrenamiento individual de MobileNetV3-Small.

Se usa como alternativa liviana para comparar precisión y latencia frente
a EfficientNet-B0 y CNN Baseline.
"""
from _run_single_model import run_single_model

if __name__ == '__main__':
    # Arquitectura liviana orientada a baja latencia.
    run_single_model('mobilenetv3_small')
