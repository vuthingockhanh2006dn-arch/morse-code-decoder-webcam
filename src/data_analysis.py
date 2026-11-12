"""Thống kê dữ liệu thô: số mẫu mỗi lớp, số người, số phiên, ánh sáng, góc quay, khoảng cách, nền.

Chạy:
    python3 src/data_analysis.py

Ghi ra reports/dataset_statistics.json và các hình trong reports/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CONDITION_COLUMNS,
    FIGURES_DIR,
    GESTURE_COLUMN,
    GESTURE_LABELS,
    POSE_LABELS,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from dataset_io import load_raw_dataset  # noqa: E402


def counts(df: pd.DataFrame, column: str) -> Dict[str, int]:
    if column not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[column].value_counts().items()}


def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    session_id = df["subject"].astype(str) + "/" + df["session"].astype(str)
    gesture_counts = {g: counts(df, GESTURE_COLUMN).get(g, 0) for g in GESTURE_LABELS}
    pose_counts = {p: counts(df, TARGET_COLUMN).get(p, 0) for p in POSE_LABELS}
    nonzero = [v for v in pose_counts.values() if v > 0]
    return {
        "total_samples": int(len(df)),
        "subjects": int(df["subject"].nunique()),
        "sessions": int(session_id.nunique()),
        "sessions_per_subject": {str(k): int(v) for k, v in df.groupby("subject")["session"].nunique().items()},
        "gesture_counts": gesture_counts,
        "pose_counts": pose_counts,
        "pose_imbalance_max_over_min": round(max(nonzero) / min(nonzero), 2) if nonzero else None,
        **{c: counts(df, c) for c in CONDITION_COLUMNS},
    }


def plot(df: pd.DataFrame, stats: Dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chưa cài matplotlib, bỏ qua phần vẽ hình.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for key, title, name in (
        ("gesture_counts", "So mau moi nhan cu chi", "gesture_distribution.png"),
        ("pose_counts", "So mau moi nhan tu the (nhan huan luyen)", "pose_distribution.png"),
    ):
        ax = pd.Series(stats[key]).plot(kind="bar", figsize=(8, 4), color="#2b5c8f", edgecolor="black", rot=0)
        ax.set_title(title)
        ax.set_ylabel("So mau")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / name, dpi=150)
        plt.close()

    available = [c for c in CONDITION_COLUMNS if stats.get(c)]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4), squeeze=False)
        for ax, col in zip(axes[0], available):
            pd.Series(stats[col]).plot(kind="bar", ax=ax, edgecolor="black", rot=15)
            ax.set_title(col)
            ax.set_ylabel("So mau")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "condition_distribution.png", dpi=150)
        plt.close()
    print(f"Hình đã lưu trong {FIGURES_DIR}")


def main() -> None:
    df = load_raw_dataset()
    stats = analyze(df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "dataset_statistics.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Tổng số mẫu: {stats['total_samples']}")
    print(f"Số người: {stats['subjects']} | Số phiên: {stats['sessions']}")
    print("Mẫu theo nhãn cử chỉ:", stats["gesture_counts"])
    print("Mẫu theo nhãn tư thế:", stats["pose_counts"], "| tỷ lệ lệch max/min:", stats["pose_imbalance_max_over_min"])
    for col in CONDITION_COLUMNS:
        print(f"{col}:", stats.get(col))
    plot(df, stats)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
