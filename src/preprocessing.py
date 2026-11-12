"""Làm sạch dữ liệu thô, tạo 66 đặc trưng và chia train/val/test.

Các bước:
  1. Gộp data/raw/**/*.csv.
  2. Bỏ dòng thiếu giá trị, nhãn lạ, dòng trùng, tay nằm ngoài khung hình, tay lỗi.
  3. Chuẩn hóa landmark và thêm 3 khoảng cách Euclid (feature_engineering.py).
  4. Chia 70/15/15 theo phiên quay (mặc định) hoặc theo người.

Chạy:
    python3 src/preprocessing.py                       # chia theo phiên
    python3 src/preprocessing.py --split-by subject    # chia theo người
    python3 src/preprocessing.py --allow-frame-split   # chỉ để chạy thử khi chưa đủ phiên
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CONDITION_COLUMNS,
    FEATURE_COLUMNS,
    FRAME_MARGIN,
    GESTURE_COLUMN,
    LANDMARK_COLUMNS,
    POSE_LABELS,
    PROCESSED_DIR,
    SPLIT_NAMES,
    SPLIT_RATIOS,
    TARGET_COLUMN,
)
from dataset_io import load_raw_dataset  # noqa: E402
from feature_engineering import extract_full_feature_vector, is_fully_visible  # noqa: E402
from splitting import assign_groups_to_splits  # noqa: E402

ID_COLUMNS = ["subject", "session", GESTURE_COLUMN, TARGET_COLUMN]


def clean_and_extract(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    required = ID_COLUMNS + ["handedness"] + LANDMARK_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Dữ liệu thô thiếu cột: {missing[:5]}. Thu lại bằng data_collection.py mới.")

    report = {"raw_rows": len(df)}
    df = df.dropna(subset=required)
    report["after_dropna"] = len(df)

    df = df[df[TARGET_COLUMN].isin(POSE_LABELS)]
    report["after_unknown_label_removed"] = len(df)

    df = df.drop_duplicates(subset=LANDMARK_COLUMNS + [GESTURE_COLUMN]).reset_index(drop=True)
    report["after_dedup"] = len(df)

    landmarks = df[LANDMARK_COLUMNS].to_numpy(dtype=float)
    hands = df["handedness"].tolist()
    features, keep = [], []
    out_of_frame = degenerate = 0
    for i in range(len(df)):
        pts = landmarks[i].tolist()
        if not is_fully_visible(pts, FRAME_MARGIN):
            out_of_frame += 1
            continue
        try:
            features.append(extract_full_feature_vector(pts, hands[i]))
        except ValueError:
            degenerate += 1
            continue
        keep.append(i)
    report["removed_out_of_frame"] = out_of_frame
    report["removed_degenerate_hand"] = degenerate
    report["final_rows"] = len(keep)
    if not keep:
        raise SystemExit("Không còn mẫu hợp lệ sau khi làm sạch.")

    meta_cols = [c for c in ID_COLUMNS + CONDITION_COLUMNS if c in df.columns]
    out = pd.concat(
        [df.loc[keep, meta_cols].reset_index(drop=True), pd.DataFrame(features, columns=FEATURE_COLUMNS)],
        axis=1,
    )
    return out, report


def split_dataset(df: pd.DataFrame, split_by: str, seed: int, allow_frame_split: bool):
    subject = df["subject"].astype(str)
    key = subject if split_by == "subject" else subject + "/" + df["session"].astype(str)
    try:
        assignment = assign_groups_to_splits(key.value_counts().to_dict(), SPLIT_RATIOS, seed)
    except ValueError as err:
        if not allow_frame_split:
            unit = "người" if split_by == "subject" else "phiên"
            raise SystemExit(
                f"{err}.\nCần thu thêm {unit} (khuyến nghị), hoặc thêm --allow-frame-split để chạy thử. "
                "Chia theo khung hình làm điểm test cao hơn thực tế vì các khung gần giống nhau lọt vào cả hai tập."
            )
        print("CẢNH BÁO: chia ngẫu nhiên theo khung hình, điểm test sẽ cao giả tạo.")
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_train = int(SPLIT_RATIOS[0] * len(df))
        n_val = int(SPLIT_RATIOS[1] * len(df))
        return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:], {}

    which = key.map(assignment)
    parts = [df[which == name].reset_index(drop=True) for name in SPLIT_NAMES]
    members = {name: sorted(g for g, s in assignment.items() if s == name) for name in SPLIT_NAMES}
    return (*parts, members)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý và chia tập dữ liệu")
    parser.add_argument("--split-by", choices=["session", "subject"], default="session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-frame-split", action="store_true")
    args = parser.parse_args()

    df, report = clean_and_extract(load_raw_dataset())
    print("Làm sạch:", json.dumps(report, ensure_ascii=False))

    train, val, test, members = split_dataset(df, args.split_by, args.seed, args.allow_frame_split)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in zip(SPLIT_NAMES, (train, val, test)):
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        share = len(part) / len(df) * 100
        print(f"{name:>5}: {len(part):6d} mẫu ({share:4.1f}%)  {members.get(name, '')}")
        absent = sorted(set(POSE_LABELS) - set(part[TARGET_COLUMN]))
        if absent:
            print(f"       CẢNH BÁO: tập {name} thiếu hẳn tư thế {absent}. Mỗi phiên nên thu đủ cả 6 nhãn.")

    (PROCESSED_DIR / "cleaning_report.json").write_text(
        json.dumps({**report, "split_by": args.split_by, "split_members": members}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Đã lưu vào {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
