#!/usr/bin/env python3
from pathlib import Path
import runpy
print('Este script ahora delega en 00_run_dataset_eda_pipeline.py')
runpy.run_path(str(Path(__file__).with_name('00_run_dataset_eda_pipeline.py')), run_name='__main__')
