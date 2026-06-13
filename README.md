# FaceGuard: Sistema Web de Autenticación Facial con Detección de Vida Anti-Spoofing

## 1. Descripción general

**FaceGuard** es un sistema web local de autenticación facial con detección de vida (*liveness detection*) y clasificación anti-spoofing. El sistema recibe un crop facial desde la webcam, ejecuta inferencia en un backend FastAPI y decide si el rostro corresponde a una persona real o a un ataque de presentación.

El proyecto cubre el flujo completo:

```text
Datasets públicos
↓
EDA + limpieza + etiquetado
↓
Procesamiento RGB y Depth
↓
Entrenamiento de modelos CNN / EfficientNet / MobileNet / CDCN
↓
Evaluación biométrica: APCER, BPCER, ACER, ROC-AUC
↓
Consolidación de resultados y gráficos
↓
Integración de checkpoints .pt en backend
↓
Prueba desde cliente web local
```

El desarrollo local se realiza en **Visual Studio Code**. El entrenamiento formal se ejecuta en **Google Colab con GPU L4**.

---

## 2. Objetivo

Construir un prototipo funcional capaz de clasificar una entrada facial como:

```text
LIVE  = 1
SPOOF = 0
```

Escenarios considerados:

```text
- rostro real frente a cámara;
- fotografía impresa;
- imagen mostrada en pantalla;
- video replay;
- recorte de fotografía;
- baja iluminación;
- rostro parcialmente visible.
```

---

## 3. Arquitecturas entrenadas

El proyecto entrena cuatro arquitecturas de manera separada.

### 3.1 CNN Baseline

Modelo convolucional propio entrenado desde cero. Funciona como línea base experimental.

### 3.2 EfficientNet-B0

Modelo con transferencia de aprendizaje. Es el modelo recomendado para la demo web RGB porque obtuvo el mejor equilibrio entre ACER, ROC-AUC y latencia entre los modelos RGB.

### 3.3 MobileNetV3-Small

Modelo liviano orientado a inferencia rápida y despliegue eficiente.

### 3.4 CDCN

Central Difference Convolutional Network. Modelo especializado en anti-spoofing, diseñado para capturar patrones finos de textura relacionados con ataques de presentación.

---

## 4. Estructura del proyecto

```text
faceguard-project/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   ├── core/
│   │   ├── database/
│   │   ├── models/
│   │   │   ├── architectures/
│   │   │   └── weights/
│   │   ├── services/
│   │   ├── utils/
│   │   ├── config.py
│   │   └── main.py
│   ├── requirements.txt
│   └── run.py
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── face/
│       ├── services/
│       ├── styles/
│       └── main.js
│
├── training/
│   ├── configs/
│   ├── requirements.txt
│   └── scripts/
│       ├── 00_run_dataset_eda_pipeline.py
│       ├── 08_train_all_models.py
│       ├── 12_train_cnn_baseline.py
│       ├── 13_train_efficientnet_b0.py
│       ├── 14_train_mobilenetv3_small.py
│       ├── 15_train_cdcn.py
│       ├── 16_consolidate_model_results.py
│       └── 17_generate_training_figures.py
│
├── scripts/
│   ├── colab_train.sh
│   ├── run_backend.sh
│   └── run_frontend.sh
│
├── docs/
├── experiments/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 5. Datasets utilizados

Los datasets se descargan desde Kaggle y no se suben a GitHub.

### 5.1 CASIA-FASD

URL:

```text
https://www.kaggle.com/datasets/minhnh2107/casiafasd
```

Uso:

```text
RGB   → imágenes de color
Depth → mapas de profundidad
```

Mapeo de etiquetas:

```text
*_real.jpg → LIVE  → 1
*_fake.jpg → SPOOF → 0
```

### 5.2 Anti-Spoofing Dataset

URL:

```text
https://www.kaggle.com/datasets/tapakah68/anti-spoofing
```

Mapeo de etiquetas:

```text
live_selfie       → LIVE  → 1
live_video        → LIVE  → 1
cut-out printouts → SPOOF → 0
printouts         → SPOOF → 0
replay            → SPOOF → 0
```

---

## 6. Preparación local

Crear entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar dependencias de backend:

```bash
pip install -r backend/requirements.txt
```

Instalar dependencias de entrenamiento:

```bash
pip install -r training/requirements.txt
```

---

## 7. Ubicación local de datasets

Para ejecutar EDA local, coloca los datasets en:

```text
training/data/raw/
├── casia_fasd/
│   ├── color/                 # o estructura descargada desde Kaggle
│   └── depth/
└── anti_spoofing/
    ├── cut-out printouts/
    ├── live_selfie/
    ├── live_video/
    ├── printouts/
    ├── replay/
    └── anti-spoofing.csv
```

El pipeline detecta recursivamente imágenes y videos dentro de esas carpetas.

---

## 8. EDA y preparación de datos

Ejecutar desde la raíz:

```bash
python training/scripts/00_run_dataset_eda_pipeline.py
```

El pipeline realiza:

```text
1. indexación de archivos;
2. construcción de metadata_raw.csv;
3. validación de etiquetas;
4. EDA inicial;
5. limpieza de datos;
6. split train/val/test;
7. detección y recorte de rostros RGB;
8. recorte depth usando el bounding box RGB correspondiente;
9. extracción de frames desde videos;
10. preparación de training/data/processed/;
11. preparación de training/data/processed_depth/;
12. generación de reportes.
```

Salidas principales:

```text
training/data/metadata/metadata_raw.csv
training/data/metadata/metadata_clean.csv
training/data/metadata/metadata_processed.csv
training/data/metadata/train.csv
training/data/metadata/val.csv
training/data/metadata/test.csv
training/data/metadata/casia_rgb_depth_pairs.csv
training/outputs/eda/
training/outputs/reports/dataset_eda_pipeline_report.md
```

---

## 9. Entrenamiento local por modelo

CNN Baseline:

```bash
python training/scripts/12_train_cnn_baseline.py --modality rgb --epochs 20 --batch-size 32
```

EfficientNet-B0:

```bash
python training/scripts/13_train_efficientnet_b0.py --modality rgb --epochs 20 --batch-size 32
```

MobileNetV3-Small:

```bash
python training/scripts/14_train_mobilenetv3_small.py --modality rgb --epochs 20 --batch-size 32
```

CDCN:

```bash
python training/scripts/15_train_cdcn.py --modality rgb --epochs 20 --batch-size 32
```

Entrenamiento RGB + Depth:

```bash
python training/scripts/08_train_all_models.py --models all --modality all --epochs 20 --batch-size 32
```

---

## 10. Consolidación y gráficos

Consolidar resultados:

```bash
python training/scripts/16_consolidate_model_results.py
```

Generar gráficos para informe:

```bash
python training/scripts/17_generate_training_figures.py
```

Archivos generados:

```text
training/outputs/reports/model_comparison.csv
training/outputs/reports/model_comparison.md
training/outputs/reports/best_model.json
training/outputs/reports/figures_summary.md
training/outputs/figures/
```

Gráficos recomendados para el informe:

```text
comparison_acer_rgb.png
comparison_roc_auc_rgb.png
comparison_accuracy_rgb.png
comparison_latency_rgb.png
best_rgb_model_profile.png
efficientnet_b0_rgb_loss_curve.png
cdcn_rgb_loss_curve.png
```

---

## 11. Entrenamiento en Google Colab con GPU L4

### 11.1 Clonar proyecto

```bash
!git clone https://github.com/manadevelop/faceguard-project.git
%cd faceguard-project
```

### 11.2 Montar Google Drive

Ejecutar en una celda de Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

El bash no monta Drive internamente; solo valida que Drive ya esté montado. Esto evita errores de kernel en Colab.

### 11.3 Configurar token Kaggle

En `scripts/colab_train.sh`, reemplazar temporalmente:

```bash
KAGGLE_API_TOKEN="INGRESE_AQUI_SU_TOKEN_KAGGLE"
```

por el token real:

```bash
KAGGLE_API_TOKEN="KGAT_TU_TOKEN_REAL"
```

### 11.4 Prueba rápida

```bash
!MODALITY=rgb EPOCHS=1 BATCH_SIZE=16 bash scripts/colab_train.sh
```

### 11.5 Entrenamiento formal

```bash
!MODALITY=all EPOCHS=20 BATCH_SIZE=32 bash scripts/colab_train.sh
```

---

## 12. Estructura generada en Google Drive

Datasets cacheados:

```text
/content/drive/MyDrive/faceguard_data/raw/casia_fasd/
/content/drive/MyDrive/faceguard_data/raw/anti_spoofing/
```

Resultados:

```text
/content/drive/MyDrive/faceguard_runs/faceguard_YYYYMMDD_HHMMSS/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── reports/
├── metadata/
├── dataset_summary/
└── run_config.txt
```

---

## 13. Resultados del entrenamiento formal

La corrida formal usada para la integración fue:

```text
MODALITY=all
EPOCHS=20
BATCH_SIZE=32
```

Run de referencia:

```text
/content/drive/MyDrive/faceguard_runs/faceguard_20260609_231006
```

Resultados RGB principales:

| Modelo | Modalidad | Accuracy | ROC-AUC | APCER | BPCER | ACER | Threshold | Latencia ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | RGB | 0.9692 | 0.9951 | 0.0114 | 0.0847 | **0.0480** | 0.725 | 1.212 |
| CNN Baseline | RGB | 0.9678 | **0.9961** | 0.0114 | 0.0899 | 0.0507 | 0.640 | **1.124** |
| MobileNetV3-Small | RGB | 0.9440 | 0.9871 | 0.0514 | 0.0688 | 0.0601 | 0.735 | 1.172 |
| CDCN | RGB | 0.9566 | 0.9891 | 0.0210 | 0.1058 | 0.0634 | 0.895 | 4.741 |

Modelo recomendado para demo web RGB:

```text
EfficientNet-B0 RGB
Checkpoint: efficientnet_b0_rgb_best.pt
Threshold: 0.725
```

Los modelos Depth obtuvieron resultados perfectos en el conjunto de prueba depth. Sin embargo, la demo web usa cámara RGB convencional, por lo que el modelo práctico para producción web es un checkpoint RGB.

---

## 14. Integración de checkpoints en backend

El backend carga checkpoints desde:

```text
backend/app/models/weights/
```

Nombres esperados:

```text
backend/app/models/weights/cnn_baseline_rgb_best.pt
backend/app/models/weights/efficientnet_b0_rgb_best.pt
backend/app/models/weights/mobilenetv3_small_rgb_best.pt
backend/app/models/weights/cdcn_rgb_best.pt
```

Para la demo principal, colocar:

```text
backend/app/models/weights/efficientnet_b0_rgb_best.pt
```

Los `.pt` no se suben a GitHub.

---

## 15. Backend FastAPI

Ejecutar desde la raíz:

```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend
```

Endpoints principales:

```text
GET  /api/v1/health
GET  /api/v1/liveness/models
POST /api/v1/liveness/image
POST /api/v1/liveness/frames
POST /api/v1/identity/enroll
POST /api/v1/identity/verify
POST /api/v1/auth/verify-image
POST /api/v1/auth/verify-realtime
```

Validar modelos disponibles:

```bash
curl http://localhost:8000/api/v1/liveness/models
```

Respuesta esperada cuando los `.pt` están presentes:

```json
{
  "models": [
    {
      "model": "efficientnet_b0",
      "modality": "rgb",
      "available": true,
      "checkpoint": ".../backend/app/models/weights/efficientnet_b0_rgb_best.pt",
      "recommended_for_web": true
    }
  ]
}
```

---

## 16. Cliente web

Ejecutar backend y abrir:

```text
http://localhost:8000/faceproguard
```

Funcionalidades:

```text
- iniciar cámara;
- detener cámara;
- generar crop facial 224x224 en frontend;
- seleccionar modelo anti-spoofing;
- verificar imagen;
- verificar video/frames;
- registrar identidad local;
- visualizar respuesta JSON del backend.
```

El frontend permite seleccionar:

```text
cnn_baseline
efficientnet_b0
mobilenetv3_small
cdcn
```

---

## 17. Respuesta real del backend

Ejemplo:

```json
{
  "access_granted": true,
  "decision": "ACCESS_GRANTED",
  "liveness": {
    "is_live": true,
    "label": "REAL",
    "score": 0.931,
    "threshold": 0.725,
    "model": "efficientnet_b0",
    "modality": "rgb",
    "checkpoint": "backend/app/models/weights/efficientnet_b0_rgb_best.pt",
    "device": "cpu",
    "latency_ms": 15.42
  },
  "identity": null
}
```

---

## 18. Pruebas de demo recomendadas

```text
1. Rostro real frente a webcam.
2. Foto impresa.
3. Imagen mostrada en celular.
4. Video replay desde otro dispositivo.
5. Baja iluminación.
6. Rostro parcial.
```

Registrar resultados en:

```text
experiments/results/
```
---

## 19. Conclusión

FaceGuard implementa un flujo completo de detección de vida anti-spoofing: preparación de datasets, EDA, entrenamiento de cuatro arquitecturas, comparación biométrica, generación de gráficos, integración de checkpoints `.pt` y prueba desde un cliente web local.

Para la aplicación web se recomienda **EfficientNet-B0 RGB** por su menor ACER entre los modelos RGB y su baja latencia de inferencia.
