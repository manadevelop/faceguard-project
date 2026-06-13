#!/usr/bin/env python3
"""
FaceGuard - Entrenamiento individual de CDCN.

CDCN se incluye por su orientación específica a face anti-spoofing usando
convoluciones de diferencia central para resaltar textura y gradientes.
"""

from _run_single_model import run_single_model


# Punto de entrada del script cuando se ejecuta directamente desde consola.
if __name__ == '__main__':
    # Lanza el entrenamiento individual de CDCN usando la configuración común del pipeline.
    run_single_model('cdcn')