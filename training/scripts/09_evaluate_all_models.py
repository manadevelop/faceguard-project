#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[2]
p=root/'training'/'outputs'/'reports'/'model_comparison.csv'
if p.exists(): print(pd.read_csv(p).to_string(index=False))
else: print('No existe model_comparison.csv. Entrena primero los modelos.')
