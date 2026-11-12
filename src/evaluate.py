"""Đánh giá các mô hình đã huấn luyện trên tập test.

Ghi ra reports/:
  test_results.csv, test_results.md       Accuracy, macro Precision/Recall/F1, thời gian suy luận
  <algo>_classification_report.txt        chỉ số theo từng nhãn
  figures/<algo>_confusion_matrix.png     ma trận nhầm lẫn

Thời gian suy luận đo bằng predict_proba trên từng khung hình riêng lẻ, giống cách
dùng với webcam (cần xác suất để áp ngưỡng tin cậy).

Chạy:
    python3 src/evaluate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALGORITHMS, FIGURES_DIR, REPORTS_DIR, model_path  # noqa: E402
from dataset_io import load_processed_split  # noqa: E402


def latency_ms(model, X: np.ndarray, n: int = 300) -> float:
    idx = np.random.default_rng(0).choice(len(X), size=min(n, len(X)), replace=False)
    model.predict_proba(X[idx[:1]])  # khởi động
    start = time.perf_counter()
    for i in idx:
        model.predict_proba(X[i:i + 1])
    return (time.perf_counter() - start) / len(idx) * 1000


def plot_confusion(cm: np.ndarray, labels, title: str, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Chưa cài matplotlib/seaborn, bỏ qua hình ma trận nhầm lẫn.")
        return
    ratio = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    text = np.array([[f"{c}\n{r:.0%}" for c, r in zip(rc, rr)] for rc, rr in zip(cm, ratio)])
    plt.figure(figsize=(1.4 * len(labels) + 2, 1.2 * len(labels) + 1.5))
    sns.heatmap(ratio, annot=text, fmt="", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Du doan")
    plt.ylabel("Thuc te")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    X_test, y_test = load_processed_split("test")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for algo in ALGORITHMS:
        path = model_path(algo)
        if not path.exists():
            print(f"Bỏ qua {algo}: chưa có {path.name} (chạy train_models.py trước)")
            continue
        model = joblib.load(path)
        labels = list(model.classes_)
        pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        rows.append({
            "model": algo.upper(),
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "macro_precision": round(precision, 4),
            "macro_recall": round(recall, 4),
            "macro_f1": round(f1, 4),
            "inference_ms_per_frame": round(latency_ms(model, X_test), 2),
        })

        report = classification_report(y_test, pred, labels=labels, zero_division=0, digits=4)
        (REPORTS_DIR / f"{algo}_classification_report.txt").write_text(report, encoding="utf-8")
        print(f"\n=== {algo.upper()} ===\n{report}")

        cm = confusion_matrix(y_test, pred, labels=labels)
        plot_confusion(cm, labels, f"{algo.upper()} (test)", FIGURES_DIR / f"{algo}_confusion_matrix.png")
        confused = sorted(((cm[i, j], labels[i], labels[j]) for i in range(len(labels))
                           for j in range(len(labels)) if i != j and cm[i, j] > 0), reverse=True)[:3]
        for count, true, predicted in confused:
            print(f"  nhầm nhiều: {true} -> {predicted}: {count} mẫu")

    if not rows:
        raise SystemExit("Chưa có mô hình nào. Chạy `python3 src/train_models.py` trước.")

    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    table.to_csv(REPORTS_DIR / "test_results.csv", index=False)
    lines = ["| Mô hình | Accuracy | Macro Precision | Macro Recall | Macro F1 | Suy luận (ms/khung) |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r.model} | {r.accuracy:.4f} | {r.macro_precision:.4f} | {r.macro_recall:.4f} | "
              f"{r.macro_f1:.4f} | {r.inference_ms_per_frame:.2f} |" for r in table.itertuples()]
    (REPORTS_DIR / "test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
