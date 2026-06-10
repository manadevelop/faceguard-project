# Protocolo de selección de threshold

El threshold se selecciona en validación minimizando ACER.

Métricas usadas:

```text
APCER = FP / (FP + TN)
BPCER = FN / (FN + TP)
ACER  = (APCER + BPCER) / 2
```

El checkpoint guarda el threshold seleccionado y el backend lo usa para inferencia real.
