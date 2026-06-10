# Protocolo de limpieza de dataset

La limpieza se realiza con `training/scripts/00_run_dataset_eda_pipeline.py`.

Criterios principales:

```text
- archivo legible;
- etiqueta válida;
- rostro detectable para RGB;
- depth recortado usando bounding box RGB;
- control de baja calidad sin eliminar automáticamente todas las muestras marcadas.
```
