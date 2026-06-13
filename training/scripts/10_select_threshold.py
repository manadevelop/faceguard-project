#!/usr/bin/env python3
"""
FaceGuard - Script informativo sobre selección de threshold.

El threshold óptimo se calcula durante el entrenamiento buscando minimizar
ACER en validación. El valor queda guardado dentro de metrics_*.json y
también dentro del checkpoint .pt correspondiente.
"""
print('El threshold óptimo por ACER se selecciona durante el entrenamiento y queda en metrics_*.json')
