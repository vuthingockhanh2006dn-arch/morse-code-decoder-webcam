"""Evaluate the trained models on the held-out test set.

Outputs (in ``reports/``):
  * ``<task>_test_results.csv`` / ``.md`` - Accuracy, macro Precision/Recall/F1,
    inference time (ms per single-frame prediction) for KNN, SVM and RF.
  * ``<task>_<algo>_classification_report.txt`` - per-class metrics.
  * ``figures/<task>_<algo>_confusion_matrix.png`` - confusion matrices.

Usage::

    python src/evaluate.py --task morse
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALGORITHMS, FIGURES_DIR, REPORTS_DIR, model_path  # noqa: E402
from train_models import load_split  # noqa: E402


def inference_ms(model, X: np.ndarray, n: int = 300) -> float:
    """Mean latency of predicting ONE frame at a time, as in the live app."""
    idx = np.random.default_rng(0).choice(len(X), size=min(n, len(X)), replace=False)
    model.predict_proba(X[idx[:1]])  # warm-up
    t0 = time.perf_counter()
    for i in idx:
        model.predict_proba(X[i:i + 1])
    return (time.perf_counter() - t0) / len(idx) * 1000


def plot_confusion(cm: np.ndarray, labels, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    annot = np.array([[f"{c}\n{p:.0%}" for c, p in zip(r1, r2)] for r1, r2 in zip(cm, norm)])
    plt.figure(figsize=(1.4 * len(labels) + 2, 1.2 * len(labels) + 1.5))
    sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["morse", "sign"], default="morse")
    args = parser.parse_args()

    X_te, y_te = load_split(args.task, "test")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for algo in ALGORITHMS:
        path = model_path(args.task, algo)
        if not path.exists():
            print(f"skip {algo}: {path.name} not found (run train_models.py)")
            continue
        model = joblib.load(path)
        labels = list(model.classes_)
        pred = model.predict(X_te)
        p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
        rows.append({
            "model": algo, "accuracy": accuracy_score(y_te, pred),
            "macro_precision": p, "macro_recall": r, "macro_f1": f1,
            "inference_ms_per_frame": inference_ms(model, X_te),
        })
        report = classification_report(y_te, pred, labels=labels, zero_division=0, digits=4)
        (REPORTS_DIR / f"{args.task}_{algo}_classification_report.txt").write_text(report, encoding="utf-8")
        print(f"\n=== {algo} ===\n{report}")

        cm = confusion_matrix(y_te, pred, labels=labels)
        plot_confusion(cm, labels, f"{args.task} - {algo.upper()} (test)",
                       FIGURES_DIR / f"{args.task}_{algo}_confusion_matrix.png")
        # Most confused pairs, useful for the error analysis in the report.
        off = [(cm[i, j], labels[i], labels[j]) for i in range(len(labels))
               for j in range(len(labels)) if i != j and cm[i, j] > 0]
        for c, t, pr in sorted(off, reverse=True)[:3]:
            print(f"  confused: true {t} -> predicted {pr}: {c} samples")

    if not rows:
        raise SystemExit("No trained models found.")
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    table.to_csv(REPORTS_DIR / f"{args.task}_test_results.csv", index=False)
    md = ["| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Inference (ms/frame) |",
          "|---|---|---|---|---|---|"]
    for r in table.itertuples():
        md.append(f"| {r.model.upper()} | {r.accuracy:.4f} | {r.macro_precision:.4f} | "
                  f"{r.macro_recall:.4f} | {r.macro_f1:.4f} | {r.inference_ms_per_frame:.2f} |")
    (REPORTS_DIR / f"{args.task}_test_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
