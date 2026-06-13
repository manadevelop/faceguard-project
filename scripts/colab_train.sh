#!/usr/bin/env bash

# Detiene el script ante errores, variables no definidas o fallos en pipelines.
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

# Guarda la ruta raíz del proyecto desde el directorio actual.
PROJECT_ROOT="$(pwd)"

# ============================================================
# CONFIGURACIÓN EDITABLE
# ============================================================

# Token de Kaggle requerido para descargar datasets privados o protegidos.
KAGGLE_API_TOKEN="INGRESE_AQUI_SU_TOKEN_KAGGLE"

# Ruta principal de Google Drive dentro de Google Colab.
DRIVE_ROOT="/content/drive/MyDrive"

# Carpeta en Drive donde se cachearán los datasets descargados.
DRIVE_DATA_ROOT="${DRIVE_ROOT}/faceguard_data"

# Carpeta en Drive donde se guardarán los resultados de cada ejecución.
DRIVE_RUNS_ROOT="${DRIVE_ROOT}/faceguard_runs"

# Carpeta local del proyecto donde se copiarán los datasets raw.
LOCAL_RAW_DIR="${PROJECT_ROOT}/training/data/raw"

# Nombre único de la ejecución usando fecha y hora actual.
RUN_NAME="faceguard_$(date +%Y%m%d_%H%M%S)"

# Carpeta final de esta ejecución dentro de Google Drive.
RUN_DIR="${DRIVE_RUNS_ROOT}/${RUN_NAME}"

# Modalidad de entrenamiento: rgb, depth o all.
MODALITY="${MODALITY:-all}"

# Número de épocas de entrenamiento.
EPOCHS="${EPOCHS:-20}"

# Tamaño de batch usado durante entrenamiento.
BATCH_SIZE="${BATCH_SIZE:-32}"

# Tamaño de imagen usado para preprocesamiento y entrenamiento.
IMAGE_SIZE="${IMAGE_SIZE:-224}"

# Máximo número de frames extraídos por video.
MAX_FRAMES_PER_VIDEO="${MAX_FRAMES_PER_VIDEO:-16}"

# Identificador del dataset CASIA-FASD en Kaggle.
CASIA_DATASET="minhnh2107/casiafasd"

# Identificador del dataset Anti-Spoofing en Kaggle.
ANTI_SPOOFING_DATASET="tapakah68/anti-spoofing"

# ============================================================
# UTILIDADES
# ============================================================

# Imprime mensajes destacados para separar etapas del pipeline.
log() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

# Imprime un error controlado y detiene la ejecución.
fail() {
  echo ""
  echo "ERROR: $1"
  echo ""
  exit 1
}

# ============================================================
# 1. VALIDAR ESTRUCTURA DEL PROYECTO
# ============================================================

# Muestra la etapa actual del pipeline.
log "Validando estructura del proyecto"

# Verifica que el script se ejecute desde la raíz del proyecto.
if [ ! -d "${PROJECT_ROOT}/training" ]; then
  fail "Ejecuta este script desde la raíz del proyecto faceguard-project."
fi

# Lista de archivos obligatorios para ejecutar el pipeline completo.
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

# Recorre cada archivo requerido para validar su existencia.
for f in "${REQUIRED_FILES[@]}"; do
  # Detiene el pipeline si falta un archivo obligatorio.
  if [ ! -f "${PROJECT_ROOT}/${f}" ]; then
    fail "Falta archivo requerido: ${f}"
  fi
done

# Crea la carpeta local para datasets raw si no existe.
mkdir -p "${LOCAL_RAW_DIR}"

# Crea la carpeta de salidas de entrenamiento si no existe.
mkdir -p "${PROJECT_ROOT}/training/outputs"

# ============================================================
# 2. VALIDAR TOKEN KAGGLE
# ============================================================

# Muestra la etapa actual del pipeline.
log "Validando token de Kaggle"

# Verifica que el token de Kaggle haya sido configurado por el usuario.
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

# Muestra la etapa actual del pipeline.
log "Validando Google Drive"

# Verifica que Google Drive esté montado en Colab.
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

# Crea la carpeta raw en Drive para almacenar datasets.
mkdir -p "${DRIVE_DATA_ROOT}/raw"

# Crea la carpeta general de ejecuciones en Drive.
mkdir -p "${DRIVE_RUNS_ROOT}"

# Crea la carpeta específica de esta ejecución.
mkdir -p "${RUN_DIR}"

# ============================================================
# 4. INSTALAR DEPENDENCIAS
# ============================================================

# Muestra la etapa actual del pipeline.
log "Instalando dependencias"

# Actualiza pip dentro del entorno de Colab.
python -m pip install --upgrade pip

# Instala las dependencias Python del proyecto.
python -m pip install -r "${PROJECT_ROOT}/training/requirements.txt"

# Kaggle CLI reciente para soporte de API Tokens.
python -m pip install "kaggle>=1.8.0"

# ============================================================
# 5. CONFIGURAR KAGGLE API TOKEN
# ============================================================

# Muestra la etapa actual del pipeline.
log "Configurando Kaggle API"

# Crea la carpeta de configuración local de Kaggle.
mkdir -p "${HOME}/.kaggle"

# Kaggle CLI moderno soporta token de acceso.
# Se configura de dos formas para máxima compatibilidad:
# 1. Variable de entorno KAGGLE_API_TOKEN.
# 2. Archivo ~/.kaggle/access_token.

# Exporta el token como variable de entorno para Kaggle CLI.
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN}"

# Guarda el token en el archivo esperado por Kaggle CLI.
echo "${KAGGLE_API_TOKEN}" > "${HOME}/.kaggle/access_token"

# Restringe permisos del token para cumplir requisitos de seguridad de Kaggle.
chmod 600 "${HOME}/.kaggle/access_token"

# Muestra la versión instalada de Kaggle CLI.
kaggle --version

# Validación rápida de conexión.
echo "Validando acceso a Kaggle..."

# Verifica autenticación listando datasets relacionados a CASIA-FASD.
kaggle datasets list -s "casiafasd" | head -n 5 || fail "No se pudo autenticar con Kaggle. Revisa tu API Token."

# ============================================================
# 6. DESCARGAR / CACHEAR DATASETS EN GOOGLE DRIVE
# ============================================================

# Muestra la etapa actual del pipeline.
log "Descargando/cacheando datasets en Google Drive"

# Define la carpeta de Drive para CASIA-FASD.
CASIA_DRIVE_DIR="${DRIVE_DATA_ROOT}/raw/casia_fasd"

# Define la carpeta de Drive para Anti-Spoofing.
ANTI_DRIVE_DIR="${DRIVE_DATA_ROOT}/raw/anti_spoofing"

# Crea la carpeta de cache para CASIA-FASD.
mkdir -p "${CASIA_DRIVE_DIR}"

# Crea la carpeta de cache para Anti-Spoofing.
mkdir -p "${ANTI_DRIVE_DIR}"

# Descarga CASIA-FASD solo si la carpeta cache está vacía.
if [ -z "$(find "${CASIA_DRIVE_DIR}" -mindepth 1 -maxdepth 3 2>/dev/null | head -n 1)" ]; then
  echo "Descargando CASIA-FASD en Google Drive..."
  kaggle datasets download -d "${CASIA_DATASET}" -p "${CASIA_DRIVE_DIR}" --unzip

# Reutiliza el dataset si ya existe en Drive.
else
  echo "CASIA-FASD ya existe en Drive. Se reutiliza cache."
fi

# Descarga Anti-Spoofing solo si la carpeta cache está vacía.
if [ -z "$(find "${ANTI_DRIVE_DIR}" -mindepth 1 -maxdepth 3 2>/dev/null | head -n 1)" ]; then
  echo "Descargando Anti-Spoofing en Google Drive..."
  kaggle datasets download -d "${ANTI_SPOOFING_DATASET}" -p "${ANTI_DRIVE_DIR}" --unzip

# Reutiliza el dataset si ya existe en Drive.
else
  echo "Anti-Spoofing ya existe en Drive. Se reutiliza cache."
fi

# ============================================================
# 7. COPIAR DATASETS A DISCO LOCAL DE COLAB
# ============================================================

# Muestra la etapa actual del pipeline.
log "Copiando datasets al disco local de Colab"

# Elimina la copia local previa de CASIA-FASD.
rm -rf "${LOCAL_RAW_DIR}/casia_fasd"

# Elimina la copia local previa de Anti-Spoofing.
rm -rf "${LOCAL_RAW_DIR}/anti_spoofing"

# Crea la carpeta local para CASIA-FASD.
mkdir -p "${LOCAL_RAW_DIR}/casia_fasd"

# Crea la carpeta local para Anti-Spoofing.
mkdir -p "${LOCAL_RAW_DIR}/anti_spoofing"

# Copia CASIA-FASD desde Drive al disco local de Colab.
rsync -ah --info=progress2 "${CASIA_DRIVE_DIR}/" "${LOCAL_RAW_DIR}/casia_fasd/"

# Copia Anti-Spoofing desde Drive al disco local de Colab.
rsync -ah --info=progress2 "${ANTI_DRIVE_DIR}/" "${LOCAL_RAW_DIR}/anti_spoofing/"

# Imprime una línea en blanco antes de mostrar estructura.
echo ""

# Muestra el encabezado de la estructura local de datasets.
echo "Estructura local de datasets:"

# Lista las primeras carpetas encontradas en la estructura raw local.
find "${LOCAL_RAW_DIR}" -maxdepth 3 -type d | sort | head -n 50

# ============================================================
# 8. EDA + PREPARACIÓN DE DATASETS
# ============================================================

# Muestra la etapa actual del pipeline.
log "Ejecutando EDA + preparación de datasets"

# Ejecuta el pipeline de EDA, limpieza, detección facial, recorte y splits.
python "${PROJECT_ROOT}/training/scripts/00_run_dataset_eda_pipeline.py" \
  --image-size "${IMAGE_SIZE}" \
  --max-frames-per-video "${MAX_FRAMES_PER_VIDEO}"

# ============================================================
# 9. ENTRENAMIENTO POR ARQUITECTURA
# ============================================================

# Muestra la etapa de entrenamiento del modelo CNN Baseline.
log "Entrenando 1/4 CNN baseline"

# Entrena CNN Baseline con la modalidad y parámetros configurados.
python "${PROJECT_ROOT}/training/scripts/12_train_cnn_baseline.py" \
  --modality "${MODALITY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}"

# Muestra la etapa de entrenamiento del modelo EfficientNet-B0.
log "Entrenando 2/4 EfficientNet-B0"

# Entrena EfficientNet-B0 con la modalidad y parámetros configurados.
python "${PROJECT_ROOT}/training/scripts/13_train_efficientnet_b0.py" \
  --modality "${MODALITY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}"

# Muestra la etapa de entrenamiento del modelo MobileNetV3-Small.
log "Entrenando 3/4 MobileNetV3-Small"

# Entrena MobileNetV3-Small con la modalidad y parámetros configurados.
python "${PROJECT_ROOT}/training/scripts/14_train_mobilenetv3_small.py" \
  --modality "${MODALITY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}"

# Muestra la etapa de entrenamiento del modelo CDCN.
log "Entrenando 4/4 CDCN"

# Entrena CDCN con la modalidad y parámetros configurados.
python "${PROJECT_ROOT}/training/scripts/15_train_cdcn.py" \
  --modality "${MODALITY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}"

# ============================================================
# 10. CONSOLIDAR MÉTRICAS
# ============================================================

# Muestra la etapa actual del pipeline.
log "Consolidando métricas finales de modelos"

# Consolida los resultados generados por los modelos entrenados.
python "${PROJECT_ROOT}/training/scripts/16_consolidate_model_results.py"

# Verifica si existe el script de generación de figuras finales.
if [ -f "${PROJECT_ROOT}/training/scripts/17_generate_training_figures.py" ]; then
  log "Generando gráficos finales para informe"
  python "${PROJECT_ROOT}/training/scripts/17_generate_training_figures.py"

# Omite la generación de figuras si el script no está disponible.
else
  echo "ADVERTENCIA: No existe training/scripts/17_generate_training_figures.py"
  echo "Se omite la generación automática de gráficos."
fi

# ============================================================
# 11. GUARDAR RESULTADOS EN GOOGLE DRIVE
# ============================================================

# Muestra la etapa actual del pipeline.
log "Guardando resultados en Google Drive"

# Asegura que exista la carpeta final de esta ejecución.
mkdir -p "${RUN_DIR}"

# Copia las salidas generadas del proyecto hacia Drive.
rsync -ah "${PROJECT_ROOT}/training/outputs/" "${RUN_DIR}/outputs/"

# Copia la metadata generada hacia Drive.
rsync -ah "${PROJECT_ROOT}/training/data/metadata/" "${RUN_DIR}/metadata/"

# Genera un archivo de configuración de la corrida para trazabilidad.
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

# Verifica si existe el dataset RGB procesado.
if [ -d "${PROJECT_ROOT}/training/data/processed" ]; then
  # Guarda el número total de archivos RGB procesados.
  find "${PROJECT_ROOT}/training/data/processed" -type f | wc -l > "${RUN_DIR}/dataset_summary/processed_rgb_file_count.txt"
fi

# Verifica si existe el dataset depth procesado.
if [ -d "${PROJECT_ROOT}/training/data/processed_depth" ]; then
  # Guarda el número total de archivos depth procesados.
  find "${PROJECT_ROOT}/training/data/processed_depth" -type f | wc -l > "${RUN_DIR}/dataset_summary/processed_depth_file_count.txt"
fi

# ============================================================
# 12. FINAL
# ============================================================

# Muestra mensaje final del pipeline.
log "Entrenamiento extremo a extremo terminado correctamente"

# Informa la carpeta de resultados generada.
echo "Resultados guardados en:"

# Imprime la ruta final de la ejecución en Drive.
echo "${RUN_DIR}"

# Imprime una línea en blanco para separar la salida.
echo ""

# Muestra encabezado de archivos principales.
echo "Archivos principales:"

# Muestra la ruta del CSV comparativo de modelos.
echo "- ${RUN_DIR}/outputs/reports/model_comparison.csv"

# Muestra la ruta del reporte Markdown comparativo.
echo "- ${RUN_DIR}/outputs/reports/model_comparison.md"

# Muestra la ruta del JSON del mejor modelo.
echo "- ${RUN_DIR}/outputs/reports/best_model.json"

# Muestra la carpeta donde se guardan checkpoints.
echo "- ${RUN_DIR}/outputs/checkpoints/"

# Muestra la carpeta donde se guardan figuras.
echo "- ${RUN_DIR}/outputs/figures/"

# Muestra la carpeta donde se guarda metadata.
echo "- ${RUN_DIR}/metadata/"