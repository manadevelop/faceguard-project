#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# FaceGuard - Colab End-to-End Training Pipeline
# ============================================================
# Uso en Colab:
#
#   !git clone https://github.com/TU_USUARIO/faceguard-project.git
#   %cd faceguard-project
#   !MODALITY=rgb EPOCHS=1 BATCH_SIZE=16 bash scripts/colab_train.sh
#
# Entrenamiento formal:
#
#   !MODALITY=all EPOCHS=20 BATCH_SIZE=32 bash scripts/colab_train.sh
#
# IMPORTANTE:
# Antes de ejecutar, reemplaza la variable KAGGLE_API_TOKEN.
# No subas tu token real a GitHub.
# ============================================================

PROJECT_ROOT="$(pwd)"

# ============================================================
# CONFIGURACIÓN EDITABLE
# ============================================================

KAGGLE_API_TOKEN="INGRESE_AQUI_SU_TOKEN_KAGGLE"

DRIVE_ROOT="/content/drive/MyDrive"
DRIVE_DATA_ROOT="${DRIVE_ROOT}/faceguard_data"
DRIVE_RUNS_ROOT="${DRIVE_ROOT}/faceguard_runs"

LOCAL_RAW_DIR="${PROJECT_ROOT}/training/data/raw"

RUN_NAME="faceguard_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${DRIVE_RUNS_ROOT}/${RUN_NAME}"

MODALITY="${MODALITY:-all}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_FRAMES_PER_VIDEO="${MAX_FRAMES_PER_VIDEO:-16}"

CASIA_DATASET="minhnh2107/casiafasd"
ANTI_SPOOFING_DATASET="tapakah68/anti-spoofing"

# ============================================================
# UTILIDADES
# ============================================================

log() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

fail() {
  echo ""
  echo "ERROR: $1"
  echo ""
  exit 1
}

# ============================================================
# 1. VALIDAR ESTRUCTURA DEL PROYECTO
# ============================================================

log "Validando estructura del proyecto"

if [ ! -d "${PROJECT_ROOT}/training" ]; then
  fail "Ejecuta este script desde la raíz del proyecto faceguard-project."
fi

REQUIRED_FILES=(
  "training/scripts/00_run_dataset_eda_pipeline.py"
  "training/scripts/08_train_all_models.py"
  "training/scripts/12_train_cnn_baseline.py"
  "training/scripts/13_train_efficientnet_b0.py"
  "training/scripts/14_train_mobilenetv3_small.py"
  "training/scripts/15_train_cdcn.py"
  "training/scripts/16_consolidate_model_results.py"
  "training/scripts/17_generate_training_figures.py"
  "training/requirements.txt"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "${PROJECT_ROOT}/${f}" ]; then
    fail "Falta archivo requerido: ${f}"
  fi
done

mkdir -p "${LOCAL_RAW_DIR}"
mkdir -p "${PROJECT_ROOT}/training/outputs"

# ============================================================
# 2. VALIDAR TOKEN KAGGLE
# ============================================================

log "Validando token de Kaggle"

if [ "${KAGGLE_API_TOKEN}" = "INGRESE_AQUI_SU_TOKEN_KAGGLE" ] || [ -z "${KAGGLE_API_TOKEN}" ]; then
  echo "Debes editar el archivo scripts/colab_train.sh y reemplazar esta línea:"
  echo ""
  echo 'KAGGLE_API_TOKEN="INGRESE_AQUI_SU_TOKEN_KAGGLE"'
  echo ""
  echo "por tu token real de Kaggle, por ejemplo:"
  echo ""
  echo 'KAGGLE_API_TOKEN="KGAT_TU_TOKEN_REAL"'
  echo ""
  fail "Token Kaggle no configurado."
fi

# ============================================================
# 3. VALIDAR GOOGLE DRIVE
# ============================================================

log "Validando Google Drive"

if [ ! -d "/content/drive/MyDrive" ]; then
  echo "ERROR: Google Drive no está montado."
  echo ""
  echo "Ejecuta primero en una celda de Colab:"
  echo ""
  echo "from google.colab import drive"
  echo "drive.mount('/content/drive')"
  echo ""
  exit 1
fi

mkdir -p "${DRIVE_DATA_ROOT}/raw"
mkdir -p "${DRIVE_RUNS_ROOT}"
mkdir -p "${RUN_DIR}"

# ============================================================
# 4. INSTALAR DEPENDENCIAS
# ============================================================

log "Instalando dependencias"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/training/requirements.txt"

# Kaggle CLI reciente para soporte de API Tokens.
python -m pip install "kaggle>=1.8.0"

# ============================================================
# 5. CONFIGURAR KAGGLE API TOKEN
# ============================================================

log "Configurando Kaggle API"

mkdir -p "${HOME}/.kaggle"

# Kaggle CLI moderno soporta token de acceso.
# Se configura de dos formas para máxima compatibilidad:
# 1. Variable de entorno KAGGLE_API_TOKEN.
# 2. Archivo ~/.kaggle/access_token.
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN}"

echo "${KAGGLE_API_TOKEN}" > "${HOME}/.kaggle/access_token"
chmod 600 "${HOME}/.kaggle/access_token"

kaggle --version

# Validación rápida de conexión.
echo "Validando acceso a Kaggle..."
kaggle datasets list -s "casiafasd" | head -n 5 || fail "No se pudo autenticar con Kaggle. Revisa tu API Token."

# ============================================================
# 6. DESCARGAR / CACHEAR DATASETS EN GOOGLE DRIVE
# ============================================================

log "Descargando/cacheando datasets en Google Drive"

CASIA_DRIVE_DIR="${DRIVE_DATA_ROOT}/raw/casia_fasd"
ANTI_DRIVE_DIR="${DRIVE_DATA_ROOT}/raw/anti_spoofing"

mkdir -p "${CASIA_DRIVE_DIR}"
mkdir -p "${ANTI_DRIVE_DIR}"

if [ -z "$(find "${CASIA_DRIVE_DIR}" -mindepth 1 -maxdepth 3 2>/dev/null | head -n 1)" ]; then
  echo "Descargando CASIA-FASD en Google Drive..."
  kaggle datasets download -d "${CASIA_DATASET}" -p "${CASIA_DRIVE_DIR}" --unzip
else
  echo "CASIA-FASD ya existe en Drive. Se reutiliza cache."
fi

if [ -z "$(find "${ANTI_DRIVE_DIR}" -mindepth 1 -maxdepth 3 2>/dev/null | head -n 1)" ]; then
  echo "Descargando Anti-Spoofing en Google Drive..."
  kaggle datasets download -d "${ANTI_SPOOFING_DATASET}" -p "${ANTI_DRIVE_DIR}" --unzip
else
  echo "Anti-Spoofing ya existe en Drive. Se reutiliza cache."
fi

# ============================================================
# 7. COPIAR DATASETS A DISCO LOCAL DE COLAB
# ============================================================

log "Copiando datasets al disco local de Colab"

rm -rf "${LOCAL_RAW_DIR}/casia_fasd"
rm -rf "${LOCAL_RAW_DIR}/anti_spoofing"

mkdir -p "${LOCAL_RAW_DIR}/casia_fasd"
mkdir -p "${LOCAL_RAW_DIR}/anti_spoofing"

rsync -ah --info=progress2 "${CASIA_DRIVE_DIR}/" "${LOCAL_RAW_DIR}/casia_fasd/"
rsync -ah --info=progress2 "${ANTI_DRIVE_DIR}/" "${LOCAL_RAW_DIR}/anti_spoofing/"

echo ""
echo "Estructura local de datasets:"
find "${LOCAL_RAW_DIR}" -maxdepth 3 -type d | sort | head -n 50

# ============================================================
# 8. EDA + PREPARACIÓN DE DATASETS
# ============================================================

log "Ejecutando EDA + preparación de datasets"

python "${PROJECT_ROOT}/training/scripts/00_run_dataset_eda_pipeline.py" \
  --image-size "${IMAGE_SIZE}" \
  --max-frames-per-video "${MAX_FRAMES_PER_VIDEO}"

# ============================================================
# 9. ENTRENAMIENTO POR ARQUITECTURA
# ============================================================

log "Entrenando 1/4 CNN baseline"

python "${PROJECT_ROOT}/training/scripts/12_train_cnn_baseline.py" \
  --modality "${MODALITY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}"

log "Entrenando 2/4 EfficientNet-B0"

python "${PROJECT_ROOT}/training/scripts/13_train_efficientnet_b0.py" \
  --modality "${MODALITY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}"

log "Entrenando 3/4 MobileNetV3-Small"

python "${PROJECT_ROOT}/training/scripts/14_train_mobilenetv3_small.py" \
  --modality "${MODALITY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}"

log "Entrenando 4/4 CDCN"

python "${PROJECT_ROOT}/training/scripts/15_train_cdcn.py" \
  --modality "${MODALITY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}"

# ============================================================
# 10. CONSOLIDAR MÉTRICAS
# ============================================================

log "Consolidando métricas finales de modelos"

python "${PROJECT_ROOT}/training/scripts/16_consolidate_model_results.py"

if [ -f "${PROJECT_ROOT}/training/scripts/17_generate_training_figures.py" ]; then
  log "Generando gráficos finales para informe"
  python "${PROJECT_ROOT}/training/scripts/17_generate_training_figures.py"
else
  echo "ADVERTENCIA: No existe training/scripts/17_generate_training_figures.py"
  echo "Se omite la generación automática de gráficos."
fi

# ============================================================
# 11. GUARDAR RESULTADOS EN GOOGLE DRIVE
# ============================================================

log "Guardando resultados en Google Drive"

mkdir -p "${RUN_DIR}"

rsync -ah "${PROJECT_ROOT}/training/outputs/" "${RUN_DIR}/outputs/"
rsync -ah "${PROJECT_ROOT}/training/data/metadata/" "${RUN_DIR}/metadata/"

cat > "${RUN_DIR}/run_config.txt" <<ENDCFG
RUN_NAME=${RUN_NAME}
PROJECT_ROOT=${PROJECT_ROOT}
MODALITY=${MODALITY}
EPOCHS=${EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
IMAGE_SIZE=${IMAGE_SIZE}
MAX_FRAMES_PER_VIDEO=${MAX_FRAMES_PER_VIDEO}
CASIA_DATASET=${CASIA_DATASET}
ANTI_SPOOFING_DATASET=${ANTI_SPOOFING_DATASET}
GPU_EXPECTED=L4
ENDCFG

# Conteo rápido de archivos generados.
mkdir -p "${RUN_DIR}/dataset_summary"

if [ -d "${PROJECT_ROOT}/training/data/processed" ]; then
  find "${PROJECT_ROOT}/training/data/processed" -type f | wc -l > "${RUN_DIR}/dataset_summary/processed_rgb_file_count.txt"
fi

if [ -d "${PROJECT_ROOT}/training/data/processed_depth" ]; then
  find "${PROJECT_ROOT}/training/data/processed_depth" -type f | wc -l > "${RUN_DIR}/dataset_summary/processed_depth_file_count.txt"
fi

# ============================================================
# 12. FINAL
# ============================================================

log "Entrenamiento extremo a extremo terminado correctamente"

echo "Resultados guardados en:"
echo "${RUN_DIR}"
echo ""
echo "Archivos principales:"
echo "- ${RUN_DIR}/outputs/reports/model_comparison.csv"
echo "- ${RUN_DIR}/outputs/reports/model_comparison.md"
echo "- ${RUN_DIR}/outputs/reports/best_model.json"
echo "- ${RUN_DIR}/outputs/checkpoints/"
echo "- ${RUN_DIR}/outputs/figures/"
echo "- ${RUN_DIR}/metadata/"