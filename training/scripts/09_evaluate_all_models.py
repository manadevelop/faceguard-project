#!/usr/bin/env python3
"""
FaceGuard - Visualización rápida de resultados consolidados.

Este script no reentrena modelos. Lee el archivo model_comparison.csv
producido por el entrenamiento/consolidación y lo imprime en consola.
"""
from pathlib import Path
import pandas as pd

# Raíz del proyecto: scripts/ está dentro de training/scripts o scripts,
# por eso se utiliza parents[2] tal como en el pipeline original.
root = Path(__file__).resolve().parents[2]

# Archivo consolidado generado después del entrenamiento de modelos.
p = root / 'training' / 'outputs' / 'reports' / 'model_comparison.csv'

if p.exists():
    # Muestra la tabla completa sin índice para revisión rápida.
    print(pd.read_csv(p).to_string(index=False))
else:
    print('No existe model_comparison.csv. Entrena primero los modelos.')
