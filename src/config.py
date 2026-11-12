"""Cấu hình dùng chung: đường dẫn, nhãn, tên cột và ngưỡng chạy.

Thiết kế nhãn (xem README, chương 3):
  * Một khung hình chỉ cho biết TƯ THẾ bàn tay (đang chụm cặp ngón nào).
    Mô hình học máy phân loại 4 tư thế: ``POSE_LABELS``.
  * Chạm nhanh hay giữ, chạm một hay hai lần là chuyện của thời gian, nên do
    máy trạng thái (``fsm_controller.py``) quyết định. Kết quả là 5 sự kiện
    ``EVENTS``: DOT, DASH, CONFIRM, SPACE, DELETE.
  * Khi thu dữ liệu, người quay chọn nhãn cử chỉ (``GESTURE_LABELS``). Hệ thống
    lưu cả nhãn cử chỉ lẫn nhãn tư thế suy ra từ ``GESTURE_TO_POSE``.

File này không tạo thư mục khi import; mỗi script tự tạo thư mục nó ghi vào.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# Đường dẫn
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIG_PATH = MODELS_DIR / "runtime_config.json"

# --------------------------------------------------------------------------- #
# Nhãn
# --------------------------------------------------------------------------- #
# Nhãn cử chỉ: 5 cử chỉ quy ước + UNKNOWN (tay nghỉ hoặc tư thế ngoài quy ước).
GESTURE_LABELS: List[str] = ["DOT", "DASH", "CONFIRM", "SPACE", "DELETE", "UNKNOWN"]

# Nhãn tư thế: đây là nhãn mà mô hình học máy dự đoán trên từng khung hình.
POSE_LABELS: List[str] = ["PINCH_THUMB", "PINCH_RING", "PINCH_PINKY", "UNKNOWN"]

GESTURE_TO_POSE: Dict[str, str] = {
    "DOT": "PINCH_THUMB",       # cái + trỏ, chạm nhanh
    "DASH": "PINCH_THUMB",      # cái + trỏ, giữ
    "CONFIRM": "PINCH_RING",    # cái + áp út, chạm 2 lần
    "SPACE": "PINCH_RING",      # cái + áp út, chạm 1 lần
    "DELETE": "PINCH_PINKY",    # cái + út, chạm 1 lần
    "UNKNOWN": "UNKNOWN",
}

EVENTS: List[str] = ["DOT", "DASH", "CONFIRM", "SPACE", "DELETE"]

GESTURE_COLUMN = "gesture"
TARGET_COLUMN = "pose"  # cột nhãn dùng để huấn luyện

# --------------------------------------------------------------------------- #
# Kịch bản thu thập
# --------------------------------------------------------------------------- #
LIGHTING_CHOICES = ["daylight", "room_light", "dim_light"]
ANGLE_CHOICES = ["front_0deg", "angle_30deg"]
DISTANCE_CHOICES = ["near_35cm", "mid_50cm", "far_70cm"]
BACKGROUND_CHOICES = ["plain_wall", "cluttered"]

# Ngưỡng loại mẫu lỗi khi thu (xem data_collection.py)
MIN_HAND_SCORE = 0.6   # độ tin cậy tối thiểu phân loại tay (hạ từ 0.8 để tránh loại oan khi chụm ngón)
MAX_WRIST_JUMP = 0.08  # cổ tay dịch quá mức này giữa 2 khung hình = tay quá nhanh/mờ
FRAME_MARGIN = 0.02    # landmark nằm ngoài khung hình quá mức này = tay bị cắt/che

# --------------------------------------------------------------------------- #
# Landmark MediaPipe (21 điểm) và đặc trưng
# --------------------------------------------------------------------------- #
N_LANDMARKS = 21
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
RING_TIP = 16
PINKY_TIP = 20

# 63 giá trị: x0, y0, z0, ..., x20, y20, z20
LANDMARK_COLUMNS: List[str] = [f"{axis}{i}" for i in range(N_LANDMARKS) for axis in ("x", "y", "z")]

# 3 khoảng cách Euclid bổ sung (chuẩn hóa theo độ dài lòng bàn tay, lấy ngón cái làm mốc)
DISTANCE_COLUMNS: List[str] = ["dist_thumb_index", "dist_thumb_ring", "dist_thumb_pinky"]

# Vector đưa vào mô hình: 63 + 3 = 66
FEATURE_COLUMNS: List[str] = LANDMARK_COLUMNS + DISTANCE_COLUMNS

CONDITION_COLUMNS: List[str] = ["lighting", "angle", "distance", "background"]
METADATA_COLUMNS: List[str] = (
    ["subject", "session", GESTURE_COLUMN, TARGET_COLUMN, "handedness"] + CONDITION_COLUMNS + ["timestamp"]
)

# --------------------------------------------------------------------------- #
# Chia tập và mô hình
# --------------------------------------------------------------------------- #
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "val", "test")
ALGORITHMS = ("knn", "svm", "rf")


def model_path(algo: str) -> Path:
    return MODELS_DIR / f"gesture_{algo}_model.pkl"


# --------------------------------------------------------------------------- #
# Tham số chạy của máy trạng thái (có thể chỉnh trong models/runtime_config.json)
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    "noise_time": 0.05,           # chạm ngắn hơn mức này (giây) bị bỏ qua
    "dash_time": 0.40,            # giữ >= mức này = DASH, ngắn hơn = DOT
    "double_tap_window": 0.45,    # lần chạm thứ 2 phải đến trong khoảng này = CONFIRM
    "confidence_threshold": 0.75, # xác suất thấp hơn mức này coi như UNKNOWN
    "smoothing_window": 5,        # số khung hình bỏ phiếu đa số
    "release_frames": 3,          # số khung hình liên tiếp để tính là đã nhả/đã nghỉ
    "best_model": None,           # train_models.py ghi vào đây
}


def load_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
