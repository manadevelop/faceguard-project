#!/usr/bin/env python3
"""
FaceGuard - Entrenamiento individual de CDCN.

CDCN se incluye por su orientación específica a face anti-spoofing usando
convoluciones de diferencia central para resaltar textura y gradientes.
"""
from _run_single_model import run_single_model

if __name__ == '__main__':
    # Arquitectura especializada en anti-spoofing facial.
    run_single_model('cdcn')
