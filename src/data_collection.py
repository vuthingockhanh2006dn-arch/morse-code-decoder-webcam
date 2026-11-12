"""Thu dữ liệu cử chỉ qua webcam, lưu landmark (không lưu ảnh hay video).

Mỗi lần chạy là một phiên: ``data/raw/<subject>/<session>.csv``. Mỗi phiên nên cố định
một bộ điều kiện (ánh sáng, góc, khoảng cách, nền) và thu đủ cả 6 nhãn.

Ví dụ:
    python3 src/data_collection.py --subject user_01 --session s01 \
        --lighting room_light --angle front_0deg --distance mid_50cm --background plain_wall

Phím:
    0 UNKNOWN (tay nghỉ hoặc tư thế ngoài quy ước)   1 DOT       2 DASH
    3 CONFIRM                                         4 SPACE     5 DELETE
    SPACE: tạm dừng    u: xóa đợt ghi gần nhất    q hoặc ESC: lưu và thoát

Cách thực hiện: bấm phím nhãn rồi GIỮ ỔN ĐỊNH tư thế cặp ngón tương ứng, hơi xoay/nghiêng
nhẹ giữa chừng. Mô hình chỉ học tư thế của từng khung hình, nên với DOT và DASH hãy giữ
chụm ngón thay vì gõ nhịp; việc tách chạm nhanh/giữ do máy trạng thái đảm nhiệm.

Khung hình bị loại và không được ghi:
    * không thấy tay, hoặc độ tin cậy MediaPipe thấp hơn --min-score
    * có landmark nằm ngoài khung hình (tay bị cắt hoặc che)
    * cổ tay dịch quá nhanh so với khung trước (tay mờ do chuyển động)
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    ANGLE_CHOICES,
    BACKGROUND_CHOICES,
    DISTANCE_CHOICES,
    FRAME_MARGIN,
    GESTURE_TO_POSE,
    LANDMARK_COLUMNS,
    LIGHTING_CHOICES,
    MAX_WRIST_JUMP,
    METADATA_COLUMNS,
    MIN_HAND_SCORE,
    RAW_DIR,
)
from feature_engineering import is_fully_visible  # noqa: E402
from hand_tracker import HandTracker  # noqa: E402

KEY_ORDER = ["UNKNOWN", "DOT", "DASH", "CONFIRM", "SPACE", "DELETE"]  # phím 0..5

HINTS = {  # chỉ dùng ASCII vì cv2.putText không vẽ được dấu tiếng Việt
    "UNKNOWN": "Tay nghi hoac tu the ngoai quy uoc",
    "DOT": "Giu chum ngon tro + ngon cai",
    "DASH": "Giu chum ngon tro + ngon cai",
    "CONFIRM": "Giu chum ngon tro + ngon ap ut",
    "SPACE": "Giu chum ngon tro + ngon ap ut",
    "DELETE": "Giu chum ngon tro + ngon ut",
}


def reject_reason(hand, prev_wrist: Optional[tuple], min_score: float) -> Optional[str]:
    if hand is None:
        return "no_hand"
    if hand.score < min_score:
        return "low_score"
    if not is_fully_visible(hand.landmarks, FRAME_MARGIN):
        return "out_of_frame"
    if prev_wrist is not None:
        wrist = hand.landmarks[0]
        if math.hypot(wrist[0] - prev_wrist[0], wrist[1] - prev_wrist[1]) > MAX_WRIST_JUMP:
            return "fast_motion"
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thu dữ liệu cử chỉ qua webcam",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--subject", required=True, help="Mã người tham gia, ví dụ user_01")
    p.add_argument("--session", required=True, help="Mã phiên quay, ví dụ s01")
    p.add_argument("--lighting", choices=LIGHTING_CHOICES, default="room_light")
    p.add_argument("--angle", choices=ANGLE_CHOICES, default="front_0deg")
    p.add_argument("--distance", choices=DISTANCE_CHOICES, default="mid_50cm")
    p.add_argument("--background", choices=BACKGROUND_CHOICES, default="plain_wall")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--every", type=int, default=2, help="Lưu mỗi N khung hình để bớt mẫu trùng")
    p.add_argument("--min-score", type=float, default=MIN_HAND_SCORE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = RAW_DIR / args.subject / f"{args.session}.csv"
    header = METADATA_COLUMNS + LANDMARK_COLUMNS
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            if next(csv.reader(f), None) != header:
                raise SystemExit(f"{out_path} có cấu trúc cột cũ. Đổi tên --session hoặc xóa file đó.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được webcam số {args.camera}")
    tracker = HandTracker()

    rows, bursts = [], []
    counts = {g: 0 for g in KEY_ORDER}
    rejects: dict = {}
    active, frame_no, prev_wrist = None, 0, None

    print(f"Người: {args.subject} | Phiên: {args.session} | {args.lighting} | {args.angle} | "
          f"{args.distance} | {args.background}\nGhi vào: {out_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # chế độ gương; khi dùng thật cũng phải lật như vậy
            hand = tracker.process(frame)
            frame_no += 1

            reason = reject_reason(hand, prev_wrist, args.min_score)
            prev_wrist = tuple(hand.landmarks[0][:2]) if hand is not None else None
            if hand is not None:
                tracker.draw(frame, hand)

            if active is not None and frame_no % args.every == 0:
                if reason is None:
                    rows.append([args.subject, args.session, active, GESTURE_TO_POSE[active], hand.handedness,
                                 args.lighting, args.angle, args.distance, args.background,
                                 round(time.time(), 3), *hand.landmarks.reshape(-1).tolist()])
                    counts[active] += 1
                else:
                    rejects[reason] = rejects.get(reason, 0) + 1

            status = f"GHI: {active}" if active else "TAM DUNG (bam 0-5)"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if active else (200, 200, 200), 2)
            if active and reason:
                cv2.putText(frame, f"Bo qua: {reason}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            for i, g in enumerate(KEY_ORDER):
                cv2.putText(frame, f"[{i}] {g}: {counts[g]}", (10, 90 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0) if g == active else (220, 220, 220), 1)
            h = frame.shape[0]
            cv2.putText(frame, f"{args.lighting} | {args.angle} | {args.distance} | {args.background}",
                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if active:
                cv2.putText(frame, HINTS[active], (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Thu du lieu cu chi", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                active = None
            elif key == ord("u") and bursts:
                start = bursts.pop()
                for r in rows[start:]:
                    counts[r[2]] -= 1
                del rows[start:]
                active = None
            elif ord("0") <= key <= ord("5"):
                active = KEY_ORDER[key - ord("0")]
                bursts.append(len(rows))
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    if not rows:
        print("Không ghi được mẫu nào.")
        return
    is_new = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerows(rows)
    print(f"Đã lưu {len(rows)} mẫu vào {out_path}")
    print("Số mẫu mỗi nhãn:", counts)
    print("Khung hình bị loại:", rejects or "không có")


if __name__ == "__main__":
    main()
