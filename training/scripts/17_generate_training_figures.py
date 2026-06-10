from pathlib import Path
import json
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "training" / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(v, default=None):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def annotate_bars(ax, values, fmt="{:.4f}"):
    for i, v in enumerate(values):
        if v is None or pd.isna(v):
            continue
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)


def detect_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def load_model_comparison():
    path = REPORTS_DIR / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    df = pd.read_csv(path)

    numeric_cols = [
        "best_epoch", "selected_threshold", "accuracy", "precision",
        "recall_sensitivity", "specificity", "f1_score", "roc_auc",
        "apcer", "bpcer", "acer", "tn", "fp", "fn", "tp",
        "latency_ms_per_sample", "train_spoof_count", "train_live_count"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def generate_history_plots():
    history_files = sorted(LOGS_DIR.glob("*_history.csv"))
    if not history_files:
        print("No se encontraron archivos history.csv en training/outputs/logs")
        return

    for path in history_files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"No se pudo leer {path}: {e}")
            continue

        stem = path.stem.replace("_history", "")
        parts = stem.split("_")
        if len(parts) >= 2:
            modality = parts[-1]
            model_name = "_".join(parts[:-1])
        else:
            model_name = stem
            modality = "unknown"

        epoch_col = detect_col(df, ["epoch"])
        train_loss_col = detect_col(df, ["train_loss"])
        val_loss_col = detect_col(df, ["val_loss"])
        val_auc_col = detect_col(df, ["val_auc", "roc_auc"])
        val_acer_col = detect_col(df, ["val_acer", "acer"])
        threshold_col = detect_col(df, ["val_threshold", "threshold", "selected_threshold"])

        if epoch_col is None:
            df["epoch"] = range(1, len(df) + 1)
            epoch_col = "epoch"

        title_prefix = f"{model_name} | {modality}".upper()

        # 1. Loss curves
        if train_loss_col or val_loss_col:
            plt.figure(figsize=(8, 5))
            if train_loss_col:
                plt.plot(df[epoch_col], df[train_loss_col], marker="o", label="Train Loss")
            if val_loss_col:
                plt.plot(df[epoch_col], df[val_loss_col], marker="o", label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss Curve - {title_prefix}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{model_name}_{modality}_loss_curve.png", dpi=200)
            plt.close()

        # 2. Val AUC
        if val_auc_col:
            plt.figure(figsize=(8, 5))
            plt.plot(df[epoch_col], df[val_auc_col], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Val ROC-AUC")
            plt.title(f"Validation ROC-AUC - {title_prefix}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{model_name}_{modality}_val_auc_curve.png", dpi=200)
            plt.close()

        # 3. Val ACER
        if val_acer_col:
            plt.figure(figsize=(8, 5))
            plt.plot(df[epoch_col], df[val_acer_col], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Val ACER")
            plt.title(f"Validation ACER - {title_prefix}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{model_name}_{modality}_val_acer_curve.png", dpi=200)
            plt.close()

        # 4. Threshold evolution
        if threshold_col:
            plt.figure(figsize=(8, 5))
            plt.plot(df[epoch_col], df[threshold_col], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Threshold")
            plt.title(f"Threshold by Epoch - {title_prefix}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{model_name}_{modality}_threshold_curve.png", dpi=200)
            plt.close()

    print("OK: gráficos de entrenamiento generados.")


def bar_plot(df, metric, title, output_name, ascending=False):
    if metric not in df.columns:
        return
    plot_df = df[["model_name", "modality", metric]].copy()
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return

    plot_df["label"] = plot_df["model_name"] + " (" + plot_df["modality"] + ")"
    plot_df = plot_df.sort_values(metric, ascending=ascending)

    plt.figure(figsize=(10, 6))
    values = plot_df[metric].tolist()
    plt.bar(plot_df["label"], values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    annotate_bars(plt.gca(), values)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=200)
    plt.close()


def generate_comparison_plots():
    df = load_model_comparison()

    # Global comparisons
    bar_plot(df, "acer", "Model Comparison - ACER (Lower is Better)", "comparison_acer_all.png", ascending=True)
    bar_plot(df, "roc_auc", "Model Comparison - ROC-AUC (Higher is Better)", "comparison_roc_auc_all.png", ascending=False)
    bar_plot(df, "accuracy", "Model Comparison - Accuracy", "comparison_accuracy_all.png", ascending=False)
    bar_plot(df, "latency_ms_per_sample", "Model Comparison - Latency ms/sample (Lower is Better)", "comparison_latency_all.png", ascending=True)

    # RGB only
    rgb_df = df[df["modality"].astype(str).str.lower() == "rgb"].copy()
    if not rgb_df.empty:
        bar_plot(rgb_df, "acer", "RGB Models - ACER (Lower is Better)", "comparison_acer_rgb.png", ascending=True)
        bar_plot(rgb_df, "roc_auc", "RGB Models - ROC-AUC (Higher is Better)", "comparison_roc_auc_rgb.png", ascending=False)
        bar_plot(rgb_df, "accuracy", "RGB Models - Accuracy", "comparison_accuracy_rgb.png", ascending=False)
        bar_plot(rgb_df, "latency_ms_per_sample", "RGB Models - Latency ms/sample (Lower is Better)", "comparison_latency_rgb.png", ascending=True)

    # Depth only
    depth_df = df[df["modality"].astype(str).str.lower() == "depth"].copy()
    if not depth_df.empty:
        bar_plot(depth_df, "acer", "Depth Models - ACER (Lower is Better)", "comparison_acer_depth.png", ascending=True)
        bar_plot(depth_df, "roc_auc", "Depth Models - ROC-AUC (Higher is Better)", "comparison_roc_auc_depth.png", ascending=False)
        bar_plot(depth_df, "accuracy", "Depth Models - Accuracy", "comparison_accuracy_depth.png", ascending=False)
        bar_plot(depth_df, "latency_ms_per_sample", "Depth Models - Latency ms/sample (Lower is Better)", "comparison_latency_depth.png", ascending=True)

    # Best RGB summary
    if not rgb_df.empty and "acer" in rgb_df.columns:
        best_rgb = rgb_df.sort_values(["acer", "roc_auc", "latency_ms_per_sample"], ascending=[True, False, True]).iloc[0]
        plt.figure(figsize=(8, 5))
        metrics = ["accuracy", "roc_auc", "f1_score", "precision", "recall_sensitivity", "specificity"]
        values = [safe_float(best_rgb.get(m), 0.0) for m in metrics]
        plt.bar(metrics, values)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Value")
        plt.title(f"Best RGB Model Profile - {best_rgb['model_name']} ({best_rgb['modality']})")
        plt.grid(True, axis="y", alpha=0.3)
        annotate_bars(plt.gca(), values)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "best_rgb_model_profile.png", dpi=200)
        plt.close()

    # Best overall summary
    if not df.empty and "acer" in df.columns:
        best_all = df.sort_values(["acer", "roc_auc", "latency_ms_per_sample"], ascending=[True, False, True]).iloc[0]
        plt.figure(figsize=(8, 5))
        metrics = ["accuracy", "roc_auc", "f1_score", "precision", "recall_sensitivity", "specificity"]
        values = [safe_float(best_all.get(m), 0.0) for m in metrics]
        plt.bar(metrics, values)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Value")
        plt.title(f"Best Overall Model Profile - {best_all['model_name']} ({best_all['modality']})")
        plt.grid(True, axis="y", alpha=0.3)
        annotate_bars(plt.gca(), values)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "best_overall_model_profile.png", dpi=200)
        plt.close()

    print("OK: gráficos comparativos generados.")


def generate_confusion_matrix_plots():
    prediction_files = sorted(REPORTS_DIR.glob("predictions_*.csv"))
    if not prediction_files:
        print("No se encontraron predictions_*.csv para matrices de confusión.")
        return

    for path in prediction_files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"No se pudo leer {path}: {e}")
            continue

        y_true_col = detect_col(df, ["y_true", "true_label", "label", "target"])
        y_pred_col = detect_col(df, ["y_pred", "pred_label", "prediction", "predicted_label"])

        if y_true_col is None or y_pred_col is None:
            print(f"Saltando matriz de confusión para {path.name}: columnas no reconocidas.")
            continue

        y_true = pd.to_numeric(df[y_true_col], errors="coerce")
        y_pred = pd.to_numeric(df[y_pred_col], errors="coerce")

        y_true = y_true.dropna().astype(int)
        y_pred = y_pred.loc[y_true.index].dropna().astype(int)

        if len(y_true) == 0 or len(y_pred) == 0:
            continue

        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

        matrix = [[tn, fp], [fn, tp]]

        plt.figure(figsize=(5, 5))
        plt.imshow(matrix)
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(f"Confusion Matrix - {path.stem.replace('predictions_', '')}")

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=12)

        plt.colorbar()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{path.stem}_confusion_matrix.png", dpi=200)
        plt.close()

    print("OK: matrices de confusión generadas.")


def generate_markdown_summary():
    df = load_model_comparison()

    best_all = None
    best_rgb = None
    best_depth = None

    if not df.empty:
        best_all = df.sort_values(["acer", "roc_auc", "latency_ms_per_sample"], ascending=[True, False, True]).iloc[0].to_dict()

        rgb_df = df[df["modality"].astype(str).str.lower() == "rgb"]
        if not rgb_df.empty:
            best_rgb = rgb_df.sort_values(["acer", "roc_auc", "latency_ms_per_sample"], ascending=[True, False, True]).iloc[0].to_dict()

        depth_df = df[df["modality"].astype(str).str.lower() == "depth"]
        if not depth_df.empty:
            best_depth = depth_df.sort_values(["acer", "roc_auc", "latency_ms_per_sample"], ascending=[True, False, True]).iloc[0].to_dict()

    md = []
    md.append("# Figures Summary\n")

    md.append("## Best Overall Model\n")
    if best_all:
        md.append(f"- Model: **{best_all['model_name']}**\n")
        md.append(f"- Modality: **{best_all['modality']}**\n")
        md.append(f"- ACER: **{best_all['acer']:.6f}**\n")
        md.append(f"- ROC-AUC: **{best_all['roc_auc']:.6f}**\n")
        md.append(f"- Accuracy: **{best_all['accuracy']:.6f}**\n")
        md.append(f"- Latency ms/sample: **{best_all['latency_ms_per_sample']:.6f}**\n")

    md.append("\n## Best RGB Model\n")
    if best_rgb:
        md.append(f"- Model: **{best_rgb['model_name']}**\n")
        md.append(f"- ACER: **{best_rgb['acer']:.6f}**\n")
        md.append(f"- ROC-AUC: **{best_rgb['roc_auc']:.6f}**\n")
        md.append(f"- Accuracy: **{best_rgb['accuracy']:.6f}**\n")
        md.append(f"- Threshold: **{best_rgb['selected_threshold']:.6f}**\n")
        md.append(f"- Checkpoint: `{best_rgb['best_checkpoint']}`\n")

    md.append("\n## Best Depth Model\n")
    if best_depth:
        md.append(f"- Model: **{best_depth['model_name']}**\n")
        md.append(f"- ACER: **{best_depth['acer']:.6f}**\n")
        md.append(f"- ROC-AUC: **{best_depth['roc_auc']:.6f}**\n")
        md.append(f"- Accuracy: **{best_depth['accuracy']:.6f}**\n")
        md.append(f"- Threshold: **{best_depth['selected_threshold']:.6f}**\n")
        md.append(f"- Checkpoint: `{best_depth['best_checkpoint']}`\n")

    md.append("\n## Generated Figures Directory\n")
    md.append(f"`{FIGURES_DIR}`\n")

    out_path = REPORTS_DIR / "figures_summary.md"
    out_path.write_text("".join(md), encoding="utf-8")
    print(f"OK: resumen markdown generado en {out_path}")


def main():
    print("Generando gráficos del entrenamiento...")
    generate_history_plots()
    generate_comparison_plots()
    generate_confusion_matrix_plots()
    generate_markdown_summary()
    print(f"LISTO. Revisa: {FIGURES_DIR}")


if __name__ == "__main__":
    main()