"""Đọc dữ liệu thô và dữ liệu đã xử lý. Mọi script dùng chung chỗ này."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, PROCESSED_DIR, RAW_DIR, TARGET_COLUMN


def load_raw_dataset() -> pd.DataFrame:
    """Gộp mọi file CSV trong data/raw/."""
    files = sorted(RAW_DIR.rglob("*.csv"))
    if not files:
        raise SystemExit(
            f"Không có file CSV nào trong {RAW_DIR}.\n"
            "Chạy `python3 src/data_collection.py --help` để bắt đầu thu dữ liệu."
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Đã đọc {len(df)} dòng từ {len(files)} file.")
    return df


def load_processed_split(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Trả về (X, y) của tập ``train``, ``val`` hoặc ``test``."""
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"Không thấy {path}. Chạy `python3 src/preprocessing.py` trước.")
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} thiếu cột {missing[:3]}... Hãy chạy lại preprocessing.py.")
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32), df[TARGET_COLUMN].astype(str).to_numpy()
