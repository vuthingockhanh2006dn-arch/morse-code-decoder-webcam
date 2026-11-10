"""Train and compare KNN, SVM and Random Forest on the 63-feature vectors.

Hyper-parameters are chosen on the validation set (macro F1). Each best model
is refit on the training set and saved to ``models/``. The algorithm with the
best validation macro F1 is recorded in ``models/scaler_config.json`` and used
by the web app. The test set is NOT touched here (see evaluate.py).

Usage::

    python src/train_models.py --task morse
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURE_COLUMNS, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, load_config, model_path, save_config  # noqa: E402

PARAM_GRIDS = {
    "knn": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    "svm": {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]},
    "rf": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
}


def build(algo: str, params: dict, seed: int) -> Pipeline:
    if algo == "knn":
        clf = KNeighborsClassifier(**params)
    elif algo == "svm":
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed, **params)
    elif algo == "rf":
        clf = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **params)
    else:
        raise ValueError(algo)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def load_split(task: str, name: str):
    path = PROCESSED_DIR / task / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run src/preprocessing.py --task {task} first.")
    df = pd.read_csv(path)
    return df[FEATURE_COLUMNS].to_numpy(), df["label"].astype(str).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["morse", "sign"], default="morse")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_tr, y_tr = load_split(args.task, "train")
    X_va, y_va = load_split(args.task, "val")
    print(f"train={len(y_tr)}  val={len(y_va)}  classes={sorted(set(y_tr))}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for algo, grid in PARAM_GRIDS.items():
        best = None
        for values in product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            model = build(algo, params, args.seed)
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            pred = model.predict(X_va)
            f1 = f1_score(y_va, pred, average="macro", zero_division=0)
            acc = accuracy_score(y_va, pred)
            print(f"  {algo:3s} {params} -> val acc={acc:.4f} macroF1={f1:.4f}")
            if best is None or f1 > best["val_macro_f1"]:
                best = {"algo": algo, "params": params, "val_accuracy": acc,
                        "val_macro_f1": f1, "fit_seconds": fit_s, "model": model}
        joblib.dump(best["model"], model_path(args.task, algo))
        print(f"* best {algo}: {best['params']} (val macroF1={best['val_macro_f1']:.4f}) "
              f"-> {model_path(args.task, algo).name}")
        rows.append({k: v for k, v in best.items() if k != "model"})

    table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / f"{args.task}_validation_results.csv"
    table.to_csv(out_csv, index=False)
    print("\nValidation summary:\n", table.to_string(index=False))

    cfg = load_config()
    cfg["best_model"][args.task] = table.iloc[0]["algo"]
    save_config(cfg)
    print(f"\nSelected '{cfg['best_model'][args.task]}' for task '{args.task}' (saved in scaler_config.json).")


if __name__ == "__main__":
    main()
