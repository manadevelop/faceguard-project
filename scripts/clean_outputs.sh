#!/usr/bin/env bash

# Detiene el script ante errores, variables no definidas o fallos en pipelines.
set -euo pipefail

# Crea la carpeta de salidas de entrenamiento si no existe.
mkdir -p training/outputs

# Elimina todos los archivos generados en training/outputs, conservando .gitkeep.
find training/outputs -type f ! -name .gitkeep -delete