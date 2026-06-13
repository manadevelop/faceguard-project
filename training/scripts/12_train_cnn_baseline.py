#!/usr/bin/env python3
"""
FaceGuard - Entrenamiento individual de CNN Baseline.

Este wrapper llama al ejecutor común _run_single_model.py para mantener
una única lógica de entrenamiento, validación, early stopping y guardado.
"""
from _run_single_model import run_single_model

if __name__ == '__main__':
    # Modelo de línea base entrenado desde cero.
    run_single_model('cnn_baseline')
