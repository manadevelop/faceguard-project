#!/usr/bin/env python3
"""
Punto de entrada para entrenar ÚNICAMENTE la CNN Baseline.

No contiene lógica propia: delega todo en _run_single_model.run_single_model(),
que parsea argumentos de línea de comandos y lanza 08_train_all_models.py
filtrando solo el modelo 'cnn_baseline'.

Uso típico:
    python training/scripts/12_train_cnn_baseline.py
    python training/scripts/12_train_cnn_baseline.py --epochs 20 --modality rgb
    python training/scripts/12_train_cnn_baseline.py --no-amp      # CPU sin AMP
    python training/scripts/12_train_cnn_baseline.py --no-pretrained  # ignorado para CNN Baseline
"""
from _run_single_model import run_single_model

if __name__ == '__main__':
    # Pasa 'cnn_baseline' como nombre de modelo; los demás args vienen de sys.argv
    run_single_model('cnn_baseline')