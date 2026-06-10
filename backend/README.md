# Backend - FaceGuard

Backend FastAPI para inferencia anti-spoofing real usando checkpoints `.pt` entrenados.

## Ejecutar

```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend
```

## Checkpoints esperados

```text
backend/app/models/weights/cnn_baseline_rgb_best.pt
backend/app/models/weights/efficientnet_b0_rgb_best.pt
backend/app/models/weights/mobilenetv3_small_rgb_best.pt
backend/app/models/weights/cdcn_rgb_best.pt
```

## Endpoints

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

## Cliente web

```text
http://localhost:8000/faceproguard
```
