"""Script kiểm thử độc lập cho các thuật toán:
1. Feature Engineering: Trích xuất 66 đặc trưng (chuẩn hóa tọa độ + 3 khoảng cách Euclid).
2. Mô hình phân loại (KNN, SVM, Random Forest): Đo độ trễ suy luận (ms), kiểm tra dự đoán.
3. Máy trạng thái (FSM) & Bộ giải mã Morse: Mô phỏng chuỗi gõ ký hiệu (DOT, DASH, CONFIRM, SPACE, DELETE).

Cách chạy:
    python3 src/test_algorithms.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Đảm bảo import được các module trong src/
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from feature_engineering import (
    compute_additional_distances,
    extract_full_feature_vector,
    normalize_landmarks,
)
from fsm_controller import MorseFSM
from morse_decoder import MorseBuffer, decode_message, decode_symbol, encode_message


# --------------------------------------------------------------------------- #
# Dữ liệu giả lập (Synthetic Landmark Samples)
# --------------------------------------------------------------------------- #
def _create_synthetic_hand(pinch: str = "NONE", scale: float = 0.2) -> List[List[float]]:
    """Tạo 21 landmark giả lập với vị trí ngón tay chuẩn hóa."""
    pts = [[0.0, 0.0, 0.0] for _ in range(21)]
    # Cổ tay tại gốc (0, 0, 0)
    pts[config.WRIST] = [0.5, 0.8, 0.0]
    # Gốc ngón giữa (định độ dài lòng bàn tay)
    pts[config.MIDDLE_MCP] = [0.5, 0.8 - scale, 0.0]

    # Các đầu ngón tay mở tự nhiên
    pts[config.THUMB_TIP] = [0.4, 0.5, 0.0]
    pts[config.INDEX_TIP] = [0.5, 0.4, 0.0]
    pts[config.RING_TIP] = [0.58, 0.45, 0.0]
    pts[config.PINKY_TIP] = [0.65, 0.5, 0.0]

    # Tùy biến tư thế chụm ngón (quy ước: ngón cái làm điểm tựa)
    if pinch == "PINCH_THUMB":
        # Ngón cái chụm đầu ngón trỏ (DOT / DASH)
        pts[config.INDEX_TIP] = list(pts[config.THUMB_TIP])
    elif pinch == "PINCH_RING":
        # Ngón cái chụm đầu ngón áp út (SPACE / CONFIRM)
        pts[config.RING_TIP] = list(pts[config.THUMB_TIP])
    elif pinch == "PINCH_PINKY":
        # Ngón cái chụm đầu ngón út (DELETE)
        pts[config.PINKY_TIP] = list(pts[config.THUMB_TIP])

    return pts


# --------------------------------------------------------------------------- #
# 1. Kiểm thử Feature Engineering
# --------------------------------------------------------------------------- #
def test_feature_engineering() -> bool:
    print("\n" + "=" * 60)
    print(" 1. KIỂM THỬ TRÍCH XUẤT ĐẶC TRƯNG (FEATURE ENGINEERING)")
    print("=" * 60)

    hand = _create_synthetic_hand("PINCH_THUMB")
    features = extract_full_feature_vector(hand, handedness="Right")

    # Kiểm tra kích thước vector (66 đặc trưng)
    assert len(features) == 66, f"Sai kích thước vector: {len(features)} != 66"
    print(f"  [OK] Kích thước vector đặc trưng: {len(features)} chiều (63 tọa độ + 3 khoảng cách Euclid).")

    # Kiểm tra tính bất biến tịnh tiến (dời cổ tay đi bất kỳ)
    hand_shifted = [[p[0] + 0.3, p[1] - 0.2, p[2] + 0.1] for p in hand]
    feat_shifted = extract_full_feature_vector(hand_shifted, handedness="Right")
    diff_shift = max(abs(a - b) for a, b in zip(features, feat_shifted))
    assert diff_shift < 1e-4, f"Không bất biến tịnh tiến: sai số {diff_shift}"
    print(f"  [OK] Bất biến tịnh tiến (Translation Invariance): sai số tối đa = {diff_shift:.2e}")

    # Kiểm tra tính bất biến co giãn (phóng to bàn tay x2)
    hand_scaled = [[(p[0] - 0.5) * 2 + 0.5, (p[1] - 0.8) * 2 + 0.8, p[2] * 2] for p in hand]
    feat_scaled = extract_full_feature_vector(hand_scaled, handedness="Right")
    diff_scale = max(abs(a - b) for a, b in zip(features, feat_scaled))
    assert diff_scale < 1e-4, f"Không bất biến co giãn: sai số {diff_scale}"
    print(f"  [OK] Bất biến tỷ lệ co giãn (Scale Invariance): sai số tối đa = {diff_scale:.2e}")

    # Kiểm tra 3 khoảng cách Euclid đặc thù
    d_thumb = compute_additional_distances(_create_synthetic_hand("PINCH_THUMB"))
    d_ring = compute_additional_distances(_create_synthetic_hand("PINCH_RING"))
    d_pinky = compute_additional_distances(_create_synthetic_hand("PINCH_PINKY"))

    assert d_thumb["dist_thumb_index"] < 0.05, "PINCH_THUMB khoảng cách cái-trỏ phải tiệm cận 0"
    assert d_ring["dist_thumb_ring"] < 0.05, "PINCH_RING khoảng cách cái-áp út phải tiệm cận 0"
    assert d_pinky["dist_thumb_pinky"] < 0.05, "PINCH_PINKY khoảng cách cái-út phải tiệm cận 0"
    print(f"  [OK] Kiểm tra 3 khoảng cách chụm: Cái-Trỏ={d_thumb['dist_thumb_index']:.3f}, Cái-Áp út={d_ring['dist_thumb_ring']:.3f}, Cái-Út={d_pinky['dist_thumb_pinky']:.3f}")

    return True


# --------------------------------------------------------------------------- #
# Bộ phân loại thuần Python (Fallback khi chưa có scikit-learn)
# --------------------------------------------------------------------------- #
class PurePythonKNN:
    """k-Nearest Neighbors thuần Python (khoảng cách Euclid)."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.X: List[List[float]] = []
        self.y: List[str] = []

    def fit(self, X: List[List[float]], y: List[str]) -> None:
        self.X = [list(x) for x in X]
        self.y = list(y)

    def predict_one(self, x: List[float]) -> str:
        distances = []
        for i, ref in enumerate(self.X):
            # Tính khoảng cách Euclid
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, ref)))
            distances.append((d, self.y[i]))
        distances.sort(key=lambda t: t[0])
        top_k = [label for _, label in distances[: self.k]]
        from collections import Counter
        return Counter(top_k).most_common(1)[0][0]

    def predict(self, X: List[List[float]]) -> List[str]:
        return [self.predict_one(x) for x in X]


class PurePythonDistanceClassifier:
    """Bộ phân loại dựa trên 3 khoảng cách Euclid ngón tay (tương đương quy tắc)."""

    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold

    def predict_one(self, x: List[float]) -> str:
        # 3 đặc trưng cuối là: dist_thumb_index, dist_index_ring, dist_index_pinky
        d_thumb, d_ring, d_pinky = x[-3], x[-2], x[-1]
        if d_thumb < self.threshold and d_thumb < min(d_ring, d_pinky):
            return "PINCH_THUMB"
        if d_ring < self.threshold and d_ring < min(d_thumb, d_pinky):
            return "PINCH_RING"
        if d_pinky < self.threshold and d_pinky < min(d_thumb, d_ring):
            return "PINCH_PINKY"
        return "UNKNOWN"

    def predict(self, X: List[List[float]]) -> List[str]:
        return [self.predict_one(x) for x in X]


# --------------------------------------------------------------------------- #
# 2. Kiểm thử và Benchmark các thuật toán học máy (KNN, SVM, Random Forest)
# --------------------------------------------------------------------------- #
def test_ml_algorithms() -> bool:
    print("\n" + "=" * 60)
    print(" 2. SO SÁNH & ĐO ĐỘ TRỄ SUY LUẬN (KNN vs SVM vs RANDOM FOREST)")
    print("=" * 60)

    # Thử nạp mô hình từ đĩa nếu đã huấn luyện
    loaded_models: Dict[str, Any] = {}
    for algo in config.ALGORITHMS:
        model_file = config.model_path(algo)
        if model_file.exists():
            try:
                import joblib
                loaded_models[algo] = joblib.load(model_file)
            except Exception as e:
                print(f"  [!] Không thể nạp {algo}: {e}")

    # Tạo tập dữ liệu giả lập 100 mẫu cho 4 tư thế
    X_dummy, y_dummy = [], []
    poses = ["PINCH_THUMB", "PINCH_RING", "PINCH_PINKY", "UNKNOWN"]
    for pose in poses:
        for _ in range(25):
            h = _create_synthetic_hand(pose if pose != "UNKNOWN" else "NONE")
            X_dummy.append(extract_full_feature_vector(h))
            y_dummy.append(pose)

    if not loaded_models:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_dummy, y_dummy)
            loaded_models["KNN (scikit-learn)"] = knn

            svm = SVC(C=1.0, kernel="rbf", probability=True)
            svm.fit(X_dummy, y_dummy)
            loaded_models["SVM (scikit-learn)"] = svm

            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X_dummy, y_dummy)
            loaded_models["RF (scikit-learn)"] = rf
        except ImportError:
            # Fallback sang bộ phân loại thuần Python
            py_knn = PurePythonKNN(k=3)
            py_knn.fit(X_dummy, y_dummy)
            loaded_models["KNN (Pure Python)"] = py_knn

            dist_clf = PurePythonDistanceClassifier(threshold=0.35)
            loaded_models["Heuristic Distance (Pure Python)"] = dist_clf

    # Đo độ trễ suy luận (Inference Latency)
    test_vec = extract_full_feature_vector(_create_synthetic_hand("PINCH_THUMB"))
    n_iters = 300

    print(f"\n  {'Thuật toán':<32} | {'Thời gian (ms/khung)':<24} | {'FPS ước tính':<12}")
    print("  " + "-" * 74)

    for algo_name, model in loaded_models.items():
        # Kiểm tra độ chính xác trên mẫu test
        pred = model.predict([test_vec])[0]
        assert pred == "PINCH_THUMB", f"{algo_name} dự đoán sai: {pred} != PINCH_THUMB"

        # Warmup
        for _ in range(5):
            _ = model.predict([test_vec])

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model.predict([test_vec])
        t1 = time.perf_counter()

        latency_ms = ((t1 - t0) / n_iters) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
        print(f"  {algo_name:<32} | {latency_ms:6.3f} ms                 | {fps:7.1f} fps")

    return True


# --------------------------------------------------------------------------- #
# 3. Kiểm thử FSM và Giải mã Morse
# --------------------------------------------------------------------------- #
def test_fsm_and_morse() -> bool:
    print("\n" + "=" * 60)
    print(" 3. KIỂM THỬ MÁY TRẠNG THÁI (FSM) & GIẢI MÃ MORSE")
    print("=" * 60)

    # 1. Kiểm tra từ điển Morse
    assert decode_symbol("...") == "S"
    assert decode_symbol("---") == "O"
    assert decode_message("... --- ...") == "SOS"
    assert encode_message("SOS") == "... --- ..."
    print("  [OK] Bảng tra Morse chuẩn xác: '... --- ...' -> 'SOS'")

    # 2. Khởi tạo FSM với cấu hình chuẩn
    cfg = config.load_config()

    fsm = MorseFSM(cfg)
    buf = MorseBuffer()

    class Clock:
        t = 0.0

    clock = Clock()
    fps = 30
    dt = 1.0 / fps

    def feed(pose: str, seconds: float) -> List[str]:
        events = []
        for _ in range(max(1, round(seconds * fps))):
            clock.t += dt
            evs = fsm.update(pose, confidence=1.0, now=clock.t)
            for e in evs:
                events.append(e)
                buf.apply(e)
        return events

    # Mô phỏng gõ chữ "E" (.) = Chạm ngón cái nhanh (< dash_time) rồi nhả
    fsm.reset()
    buf = MorseBuffer()
    clock.t = 0.0
    ev_touch = feed("PINCH_THUMB", seconds=0.15)  # ngắn hơn dash_time (0.40s)
    ev_rest = feed("UNKNOWN", seconds=0.30)
    assert "DOT" in ev_rest, f"Kỳ vọng sự kiện DOT, nhận được: {ev_touch + ev_rest}"
    assert buf.buffer == "."
    print(f"  [OK] Chạm ngắn ngón cái (0.15s) -> phát DOT: bộ đệm = '{buf.buffer}'")

    # Mô phỏng gõ chữ "T" (-) = Giữ ngón cái lâu (>= dash_time)
    fsm.reset()
    buf = MorseBuffer()
    clock.t = 0.0
    ev_hold = feed("PINCH_THUMB", seconds=0.45)   # dài hơn dash_time (0.40s)
    feed("UNKNOWN", seconds=0.30)
    assert "DASH" in ev_hold, f"Kỳ vọng sự kiện DASH, nhận được: {ev_hold}"
    assert buf.buffer == "-"
    print(f"  [OK] Giữ ngón cái (0.45s) -> phát DASH: bộ đệm = '{buf.buffer}'")

    # Mô phỏng CONFIRM (chạm 2 lần ngón áp út)
    fsm.reset()
    buf = MorseBuffer()
    clock.t = 0.0
    buf.apply("DOT")  # Giả sử đang có dấu '.'
    # Tap 1
    feed("PINCH_RING", seconds=0.10)
    feed("UNKNOWN", seconds=0.10)
    # Tap 2 (trong double_tap_window)
    ev_confirm = feed("PINCH_RING", seconds=0.10)
    feed("UNKNOWN", seconds=0.30)
    assert "CONFIRM" in ev_confirm, f"Kỳ vọng sự kiện CONFIRM, nhận được: {ev_confirm}"
    assert buf.text == "E", f"Kỳ vọng giải mã ra chữ 'E', nhận được: '{buf.text}'"
    print(f"  [OK] Chạm đúp ngón áp út -> phát CONFIRM: giải mã thành công text = '{buf.text}'")

    # Mô phỏng DELETE (chạm ngón út)
    fsm.reset()
    clock.t = 0.0
    ev_delete = feed("PINCH_PINKY", seconds=0.10)
    feed("UNKNOWN", seconds=0.30)
    assert "DELETE" in ev_delete
    assert buf.text == "", f"Kỳ vọng đã xóa text, hiện tại text = '{buf.text}'"
    print(f"  [OK] Chạm ngón út -> phát DELETE: đã xóa ký tự cuối thành công.")

    return True


# --------------------------------------------------------------------------- #
# Hàm chạy tổng thể
# --------------------------------------------------------------------------- #
def run_all_tests() -> bool:
    print("\n" + "#" * 60)
    print(" BẮT ĐẦU KIỂM THỬ TOÀN BỘ CÁC THUẬT TOÁN HỆ THỐNG MORSE")
    print("#" * 60)

    ok1 = test_feature_engineering()
    ok2 = test_ml_algorithms()
    ok3 = test_fsm_and_morse()

    print("\n" + "=" * 60)
    if ok1 and ok2 and ok3:
        print(" [HOÀN TẤT] TẤT CẢ CÁC THUẬT TOÁN ĐÃ VƯỢT QUA KIỂM THỬ 100%!")
    print("=" * 60 + "\n")
    return ok1 and ok2 and ok3


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
