# Training - FaceGuard

Esta carpeta contiene el pipeline de preparación de datos, entrenamiento, evaluación, consolidación y generación de gráficos.

## Flujo principal

```bash
python training/scripts/00_run_dataset_eda_pipeline.py
python training/scripts/12_train_cnn_baseline.py --modality rgb --epochs 20 --batch-size 32
python training/scripts/13_train_efficientnet_b0.py --modality rgb --epochs 20 --batch-size 32
python training/scripts/14_train_mobilenetv3_small.py --modality rgb --epochs 20 --batch-size 32
python training/scripts/15_train_cdcn.py --modality rgb --epochs 20 --batch-size 32
python training/scripts/16_consolidate_model_results.py
python training/scripts/17_generate_training_figures.py
```

## Entrenamiento completo RGB + Depth

```bash
python training/scripts/08_train_all_models.py --models all --modality all --epochs 20 --batch-size 32
```

## Resultados

```text
training/outputs/checkpoints/
training/outputs/logs/
training/outputs/figures/
training/outputs/reports/
```

Los resultados no se suben a GitHub.
