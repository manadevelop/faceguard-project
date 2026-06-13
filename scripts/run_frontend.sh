#!/usr/bin/env bash

# Detiene el script ante errores, variables no definidas o fallos en pipelines.
set -euo pipefail

# Informa que el frontend se sirve directamente desde el backend FastAPI.
echo "El frontend se sirve desde FastAPI: http://localhost:8000/faceproguard"