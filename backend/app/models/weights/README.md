# Pesos entrenados anti-spoofing

Coloca aquí los checkpoints `.pt` generados por el entrenamiento formal en Colab.

Para la demo web se recomienda usar el mejor modelo RGB:

```text
efficientnet_b0_rgb_best.pt
```

Ruta esperada:

```text
backend/app/models/weights/efficientnet_b0_rgb_best.pt
```

También se aceptan estas rutas/nombres:

```text
backend/app/models/weights/cnn_baseline_rgb_best.pt
backend/app/models/weights/efficientnet_b0_rgb_best.pt
backend/app/models/weights/mobilenetv3_small_rgb_best.pt
backend/app/models/weights/cdcn_rgb_best.pt

backend/app/models/weights/cnn_baseline/rgb_best.pt
backend/app/models/weights/efficientnet_b0/rgb_best.pt
backend/app/models/weights/mobilenetv3_small/rgb_best.pt
backend/app/models/weights/cdcn/rgb_best.pt
```

Los archivos `.pt`, `.pth` y `.onnx` no deben subirse a GitHub.
