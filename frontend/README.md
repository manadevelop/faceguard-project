# FaceGuard Frontend

Interfaz web local del proyecto final **FaceGuard** para detección de vida anti-spoofing y autenticación facial.

## Características

- Diseño basado en criterios de **Material Design 3**.
- Flujo separado para:
  - Verificación de liveness.
  - Enrolamiento de identidad.
- Captura de cámara en navegador.
- Recorte facial con **MediaPipe Face Detector** en `src/face/faceCropper.js`.
- Envío únicamente del crop facial normalizado a `224x224` al backend FastAPI.
- Auto-stop de cámara después de verificar o enrolar.
- Soporte responsive para escritorio y móvil.

## Flujo móvil

El modo móvil maneja tres estados:

1. **Inicial:** muestra modelo anti-spoofing y botón `Iniciar cámara` a ancho completo.
2. **Cámara activa:** muestra cámara, botón `Detener cámara` y acciones de verificación/enrolamiento a ancho completo.
3. **Captura realizada:** reemplaza la cámara por el crop facial enviado, muestra el resultado del backend y deja solo el botón `Iniciar cámara` para repetir el flujo.

## Archivos principales

```text
frontend/index.html
frontend/src/main.js
frontend/src/styles/global.css
frontend/src/face/faceCropper.js
frontend/src/services/authApi.js
```

## Uso local

El frontend se sirve desde FastAPI en:

```text
http://localhost:8000/faceproguard
```

Reiniciar backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend
```
