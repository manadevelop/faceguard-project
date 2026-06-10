#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / 'training' / 'scripts' / '08_train_all_models.py'
def run_single_model(model_name: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', default='rgb', choices=['rgb','depth','all'])
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--no-weighted-sampler', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()
    cmd = [sys.executable, str(TRAIN_SCRIPT), '--models', model_name, '--modality', args.modality, '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--image-size', str(args.image_size), '--lr', str(args.lr), '--weight-decay', str(args.weight_decay), '--num-workers', str(args.num_workers), '--seed', str(args.seed), '--patience', str(args.patience)]
    if args.no_pretrained: cmd.append('--no-pretrained')
    if args.no_weighted_sampler: cmd.append('--no-weighted-sampler')
    if args.no_amp: cmd.append('--no-amp')
    print('='*90); print(f'Entrenamiento individual: {model_name}'); print(' '.join(cmd)); print('='*90)
    subprocess.run(cmd, check=True)
__all__ = ['run_single_model']
