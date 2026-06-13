#!/usr/bin/env python3
"""
FaceGuard - Consolidación final de métricas de modelos.

Lee:
    training/outputs/reports/metrics_*_*.json

Genera:
    training/outputs/reports/model_comparison.csv
    training/outputs/reports/model_comparison.md
    training/outputs/reports/best_model.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "training" / "outputs" / "reports"


# Convierte un JSON de métricas en una fila plana para tabla comparativa.
def flatten_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    test_metrics = payload.get("test_metrics", {}) or {}
    class_counts = payload.get("class_counts_train", {}) or {}

    row = {
        "model_name": payload.get("model_name", ""),
        "modality": payload.get("modality", ""),
        "best_epoch": payload.get("best_epoch", ""),
        "selected_threshold": payload.get("threshold", ""),
        "train_spoof_count": class_counts.get("0", class_counts.get(0, "")),
        "train_live_count": class_counts.get("1", class_counts.get(1, "")),
        "best_checkpoint": payload.get("best_checkpoint", ""),
        "history_path": payload.get("history_path", ""),
        "predictions_path": payload.get("predictions_path", ""),
    }

    for key, value in test_metrics.items():
        if key != "threshold":
            row[key] = value

    return row


# Lee todos los metrics_*.json disponibles en training/outputs/reports.
def load_metrics_files() -> List[Dict[str, Any]]:
    files = sorted(REPORTS_DIR.glob("metrics_*_*.json"))
    rows: List[Dict[str, Any]] = []

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = flatten_result(payload)
            row["metrics_file"] = str(path.relative_to(PROJECT_ROOT))
            rows.append(row)
        except Exception as exc:
            rows.append({
                "model_name": "ERROR",
                "modality": "",
                "metrics_file": str(path.relative_to(PROJECT_ROOT)),
                "error": str(exc),
            })

    return rows


# Ordena resultados priorizando menor ACER, mayor ROC-AUC y menor latencia.
def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "model_name",
        "modality",
        "best_epoch",
        "selected_threshold",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
        "apcer",
        "bpcer",
        "acer",
        "tn",
        "fp",
        "fn",
        "tp",
        "latency_ms_per_sample",
        "train_spoof_count",
        "train_live_count",
        "best_checkpoint",
        "history_path",
        "predictions_path",
        "metrics_file",
    ]

    for col in preferred_cols:
        if col not in df.columns:
            df[col] = ""

    ordered_cols = preferred_cols + [c for c in df.columns if c not in preferred_cols]
    df = df.loc[:, ordered_cols].copy()

    df.loc[:, "_sort_acer"] = pd.to_numeric(df["acer"], errors="coerce")
    df.loc[:, "_sort_auc"] = pd.to_numeric(df["roc_auc"], errors="coerce")

    df = df.sort_values(
        by=["_sort_acer", "_sort_auc"],
        ascending=[True, False],
        na_position="last",
    )

    df = df.drop(columns=["_sort_acer", "_sort_auc"])

    return df

# Genera Markdown sin depender de tabulate, útil para entornos mínimos.
def dataframe_to_markdown_no_tabulate(df: pd.DataFrame, cols: List[str]) -> str:
    """
    Genera tabla Markdown sin depender de pandas.to_markdown ni tabulate.
    """
    if len(df) == 0:
        return "No se encontraron métricas.\n"

    available_cols = [c for c in cols if c in df.columns]
    table = df[available_cols].copy()

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    rows = []
    rows.append("| " + " | ".join(available_cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(available_cols)) + " |")

    for _, row in table.iterrows():
        rows.append("| " + " | ".join(fmt(row[c]) for c in available_cols) + " |")

    return "\n".join(rows) + "\n"


# Guarda un reporte Markdown legible con las columnas principales.
def save_markdown(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "model_name",
        "modality",
        "accuracy",
        "f1_score",
        "roc_auc",
        "apcer",
        "bpcer",
        "acer",
        "latency_ms_per_sample",
        "best_checkpoint",
    ]

    md = "# FaceGuard - Comparación final de modelos\n\n"
    md += dataframe_to_markdown_no_tabulate(df, cols)

    path.write_text(md, encoding="utf-8")


# Guarda best_model.json usando el criterio experimental de selección.
def save_best_model(df: pd.DataFrame, path: Path) -> None:
    if len(df) == 0:
        path.write_text("{}", encoding="utf-8")
        return

    valid = df.copy()
    valid["acer_numeric"] = pd.to_numeric(valid.get("acer", ""), errors="coerce")
    valid["auc_numeric"] = pd.to_numeric(valid.get("roc_auc", ""), errors="coerce")
    valid = valid.dropna(subset=["acer_numeric"])

    if len(valid) == 0:
        path.write_text("{}", encoding="utf-8")
        return

    valid = valid.sort_values(
        by=["acer_numeric", "auc_numeric"],
        ascending=[True, False],
    )

    best = valid.iloc[0].drop(
        labels=["acer_numeric", "auc_numeric"],
        errors="ignore",
    ).to_dict()

    path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")


# Punto de entrada: consolida resultados y escribe CSV, MD y JSON final.
def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_metrics_files()

    if len(rows) == 0:
        print("No se encontraron archivos metrics_*_*.json en:")
        print(REPORTS_DIR)
        return

    df = pd.DataFrame(rows)
    df = sort_results(df)

    comparison_csv = REPORTS_DIR / "model_comparison.csv"
    comparison_md = REPORTS_DIR / "model_comparison.md"
    best_model_json = REPORTS_DIR / "best_model.json"

    df.to_csv(comparison_csv, index=False)
    save_markdown(df, comparison_md)
    save_best_model(df, best_model_json)

    print("Consolidación completada.")
    print(f"CSV:  {comparison_csv.relative_to(PROJECT_ROOT)}")
    print(f"MD:   {comparison_md.relative_to(PROJECT_ROOT)}")
    print(f"Best: {best_model_json.relative_to(PROJECT_ROOT)}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()