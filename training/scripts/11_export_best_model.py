#!/usr/bin/env python3
from pathlib import Path
root=Path(__file__).resolve().parents[2]
ck=root/'training'/'outputs'/'checkpoints'
print('Checkpoints disponibles en:', ck)
for p in sorted(ck.rglob('*_best.pt')): print(p.relative_to(root))
