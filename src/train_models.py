"""Huấn luyện KNN, SVM, Random Forest và chọn siêu tham số trên tập validation.

Mỗi thuật toán thử một lưới tham số nhỏ; cấu hình có macro F1 cao nhất trên val được
lưu vào models/gesture_<algo>_model.pkl. Thuật toán tốt nhất được ghi vào
models/runtime_config.json. Tập test không được dùng ở đây (xem evaluate.py).

Chạy:
    python3 src/train_models.py
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALGORITHMS, MODELS_DIR, REPORTS_DIR, load_config, model_path, save_config  # noqa: E402
from dataset_io import load_processed_split  # noqa: E402

PARAM_GRIDS = {
    "knn": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    "svm": {"C": [0.1, 1.0, 10.0, 50.0], "gamma": ["scale", 0.01, 0.1]},
    "rf": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
}


def build_pipeline(algo: str, params: dict, seed: int) -> Pipeline:
    if algo == "knn":
        clf = KNeighborsClassifier(**params)
    elif algo == "svm":
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed, **params)
    elif algo == "rf":
        clf = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **params)
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: {algo}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện và so sánh KNN, SVM, Random Forest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train = load_processed_split("train")
    X_val, y_val = load_processed_split("val")
    print(f"train={len(y_train)} val={len(y_val)} đặc trưng={X_train.shape[1]} nhãn={sorted(np.unique(y_train))}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for algo in ALGORITHMS:
        grid = PARAM_GRIDS[algo]
        best = None
        print(f"\n--- {algo.upper()} ---")
        for values in product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            pipeline = build_pipeline(algo, params, args.seed)
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            pred = pipeline.predict(X_val)
            acc = accuracy_score(y_val, pred)
            f1 = f1_score(y_val, pred, average="macro", zero_division=0)
            print(f"  {params}: val accuracy={acc:.4f}  macro F1={f1:.4f}")
            if best is None or f1 > best["val_macro_f1"]:
                best = {"algo": algo, "params": str(params), "val_accuracy": acc,
                        "val_macro_f1": f1, "fit_seconds": round(fit_seconds, 3), "model": pipeline}
        joblib.dump(best["model"], model_path(algo))
        print(f"=> tốt nhất: {best['params']} (macro F1={best['val_macro_f1']:.4f}) -> {model_path(algo).name}")
        summary.append({k: v for k, v in best.items() if k != "model"})

    table = pd.DataFrame(summary).sort_values("val_macro_f1", ascending=False)
    table.to_csv(REPORTS_DIR / "validation_results.csv", index=False)
    print("\nKết quả trên validation:\n", table.to_string(index=False))

    cfg = load_config()
    cfg["best_model"] = str(table.iloc[0]["algo"])
    save_config(cfg)
    print(f"\nThuật toán tốt nhất: {cfg['best_model'].upper()} (đã ghi vào models/runtime_config.json)")


if __name__ == "__main__":
    main()
