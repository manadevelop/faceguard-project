#!/usr/bin/env python3
"""
FaceGuard - Listado de checkpoints best.pt disponibles.

Este script permite verificar qué modelos entrenados están disponibles
para ser copiados posteriormente al backend/app/models/weights/.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
ck = root / 'training' / 'outputs' / 'checkpoints'

print('Checkpoints disponibles en:', ck)

# Lista todos los checkpoints marcados como mejores por validación.
for p in sorted(ck.rglob('*_best.pt')):
    print(p.relative_to(root))
