#!/usr/bin/env python3
"""
FaceGuard - Script auxiliar para construcción de metadata.

Este archivo se mantiene para compatibilidad con el pipeline numerado.
La lógica real de construcción de metadata está integrada dentro de
00_run_dataset_eda_pipeline.py, por lo que aquí se delega la ejecución
al pipeline principal sin duplicar código.
"""
from pathlib import Path
import runpy

# Ruta al pipeline integral de preparación de datos, EDA, limpieza, crops y splits.
pipeline_path = Path(__file__).with_name('00_run_dataset_eda_pipeline.py')

print('Este script ahora delega en 00_run_dataset_eda_pipeline.py')

# Ejecuta el pipeline principal como si fuese invocado directamente por consola.
runpy.run_path(str(pipeline_path), run_name='__main__')
