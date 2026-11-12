"""Chuẩn hóa landmark và tạo đặc trưng. Chỉ dùng thư viện chuẩn của Python.

Từ 21 landmark (x, y, z) tạo vector 66 giá trị:
  * 63 giá trị: tọa độ đã dời gốc về cổ tay, lật gương nếu là tay trái, chia
    cho khoảng cách xa nhất tới cổ tay (bất biến vị trí, tay trái/phải, xa/gần).
  * 3 giá trị: khoảng cách Euclid cái-trỏ, cái-áp út, cái-út, chia cho độ dài
    lòng bàn tay (cổ tay -> gốc ngón giữa).

Đầu vào chấp nhận list lồng nhau (21, 3), list phẳng 63 phần tử, hoặc mảng numpy.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from config import (
    INDEX_TIP,
    MIDDLE_MCP,
    N_LANDMARKS,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
)

EPSILON = 1e-6


def _to_points(landmarks) -> List[List[float]]:
    """Đưa mọi dạng đầu vào về list 21 điểm [x, y, z]."""
    if hasattr(landmarks, "tolist"):
        landmarks = landmarks.tolist()
    flat: List[float] = []
    for item in landmarks:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    if len(flat) != N_LANDMARKS * 3:
        raise ValueError(f"Cần {N_LANDMARKS * 3} giá trị landmark, nhận được {len(flat)}")
    return [[float(flat[3 * i]), float(flat[3 * i + 1]), float(flat[3 * i + 2])] for i in range(N_LANDMARKS)]


def is_fully_visible(landmarks, margin: float = 0.02) -> bool:
    """False nếu có landmark nằm ngoài khung hình (tay bị cắt hoặc bị che)."""
    return all(
        -margin <= x <= 1.0 + margin and -margin <= y <= 1.0 + margin
        for x, y, _ in _to_points(landmarks)
    )


def normalize_landmarks(landmarks, handedness: Optional[str] = "Right") -> List[float]:
    """Trả về 63 giá trị đã chuẩn hóa."""
    pts = _to_points(landmarks)
    wx, wy, wz = pts[WRIST]
    shifted = []
    for x, y, z in pts:
        x, y, z = x - wx, y - wy, z - wz
        if handedness == "Left":
            x = -x
        shifted.append((x, y, z))

    scale = max(math.sqrt(x * x + y * y + z * z) for x, y, z in shifted)
    if scale < EPSILON:
        raise ValueError("Bàn tay lỗi: mọi landmark trùng nhau")
    return [value / scale for point in shifted for value in point]


def _palm_length(pts: Sequence[Sequence[float]]) -> float:
    palm = math.hypot(pts[MIDDLE_MCP][0] - pts[WRIST][0], pts[MIDDLE_MCP][1] - pts[WRIST][1])
    if palm < EPSILON:
        raise ValueError("Bàn tay lỗi: lòng bàn tay có độ dài 0")
    return palm


def compute_additional_distances(landmarks) -> Dict[str, float]:
    """3 khoảng cách Euclid 2D chuẩn hóa theo độ dài lòng bàn tay:
    - dist_thumb_index: khoảng cách ngón cái - ngón trỏ (DOT / DASH)
    - dist_thumb_ring: khoảng cách ngón cái - ngón áp út (SPACE / CONFIRM)
    - dist_thumb_pinky: khoảng cách ngón cái - ngón út (DELETE)
    """
    pts = _to_points(landmarks)
    palm = _palm_length(pts)

    def dist(a: int, b: int) -> float:
        return math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) / palm

    return {
        "dist_thumb_index": dist(THUMB_TIP, INDEX_TIP),
        "dist_thumb_ring": dist(THUMB_TIP, RING_TIP),
        "dist_thumb_pinky": dist(THUMB_TIP, PINKY_TIP),
    }


def extract_full_feature_vector(landmarks, handedness: Optional[str] = "Right") -> List[float]:
    """Vector 66 giá trị, thứ tự khớp với ``config.FEATURE_COLUMNS``."""
    d = compute_additional_distances(landmarks)
    return normalize_landmarks(landmarks, handedness) + [
        d["dist_thumb_index"],
        d["dist_thumb_ring"],
        d["dist_thumb_pinky"],
    ]
