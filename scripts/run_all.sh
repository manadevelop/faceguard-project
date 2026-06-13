#!/usr/bin/env bash

# Detiene el script ante errores, variables no definidas o fallos en pipelines.
set -euo pipefail

# Ejecuta el script principal que levanta el backend de FaceGuard.
bash scripts/run_backend.sh