# Frontend - FaceGuard

Interfaz web local servida por FastAPI en:

```text
http://localhost:8000/faceproguard
```

El navegador captura la webcam, genera un crop facial 224x224 y envía únicamente ese crop al backend.

## Funcionalidades

```text
- iniciar/detener cámara;
- seleccionar modelo anti-spoofing;
- verificar imagen;
- verificar video/frames;
- registrar identidad local;
- visualizar respuesta JSON del backend.
```

Modelos seleccionables:

```text
cnn_baseline
efficientnet_b0
mobilenetv3_small
cdcn
```
