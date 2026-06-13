#!/usr/bin/env bash

# Detiene el script ante errores, variables no definidas o fallos en pipelines.
set -euo pipefail

# Levanta el backend FastAPI usando Uvicorn con recarga automática en desarrollo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend