"""Máy trạng thái (FSM): biến chuỗi tư thế theo thời gian thành lệnh Morse.

Đầu vào mỗi khung hình: nhãn tư thế (PINCH_THUMB, PINCH_RING, PINCH_PINKY,
UNKNOWN, hoặc None nếu không thấy tay), độ tin cậy, và thời điểm (giây).
Đầu ra: danh sách sự kiện (DOT, DASH, CONFIRM, SPACE, DELETE), thường rỗng.

UNKNOWN và None đều được coi là tư thế nghỉ.

Chống nhập lặp:
  * Bỏ phiếu đa số trên N khung hình gần nhất, loại các khung nhiễu lẻ.
  * Một lần chạm chỉ tính là đã nhả sau ``release_frames`` khung liên tiếp.
  * Sau DASH, CONFIRM, DELETE máy chuyển sang LOCKED và chỉ mở khóa khi tay
    đã nghỉ ``release_frames`` khung. Giữ tay mãi cũng chỉ ra một sự kiện.

Sơ đồ:
    IDLE --PINCH_THUMB--> THUMB_DOWN --nhả, ngắn hơn dash_time--> IDLE      phát DOT
                                     --giữ từ dash_time---------> LOCKED    phát DASH
    IDLE --PINCH_RING---> RING_DOWN  --nhả---------------------> WAIT_SECOND
    WAIT_SECOND --PINCH_RING trong double_tap_window-----------> LOCKED     phát CONFIRM
    WAIT_SECOND --hết cửa sổ hoặc có tư thế chạm khác----------> IDLE       phát SPACE
    IDLE --PINCH_PINKY-----------------------------------------> LOCKED     phát DELETE
    LOCKED --tay nghỉ đủ release_frames------------------------> IDLE
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional

REST_POSES = {"UNKNOWN", "IDLE", None}
TOUCH_POSES = {"PINCH_THUMB", "PINCH_RING", "PINCH_PINKY"}


class PoseSmoother:
    """Ngưỡng tin cậy cộng bỏ phiếu đa số trên cửa sổ trượt."""

    def __init__(self, window: int = 5, confidence_threshold: float = 0.75) -> None:
        self.window: Deque[Optional[str]] = deque(maxlen=max(1, window))
        self.confidence_threshold = confidence_threshold

    def update(self, pose: Optional[str], confidence: float = 1.0) -> Optional[str]:
        if pose is not None and confidence < self.confidence_threshold:
            pose = "UNKNOWN"
        self.window.append(pose)
        winner, count = Counter(self.window).most_common(1)[0]
        # Không có đa số tuyệt đối (khung chuyển tiếp) thì coi là UNKNOWN.
        return winner if count > len(self.window) // 2 else "UNKNOWN"

    def reset(self) -> None:
        self.window.clear()


class MorseFSM:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.noise_time = float(cfg["noise_time"])
        self.dash_time = float(cfg["dash_time"])
        self.double_tap_window = float(cfg["double_tap_window"])
        self.release_frames = int(cfg["release_frames"])
        self.smoother = PoseSmoother(int(cfg["smoothing_window"]), float(cfg["confidence_threshold"]))
        self.reset()

    def reset(self) -> None:
        self.state = "IDLE"
        self.touch_start = 0.0
        self.first_tap_release = 0.0
        self._count = 0
        self._count_start: Optional[float] = None
        self.smoother.reset()

    def _go(self, state: str) -> None:
        self.state = state
        self._count = 0
        self._count_start = None

    def _released(self, pose: Optional[str], touch_pose: str, now: float) -> bool:
        """True khi ``touch_pose`` đã vắng mặt ``release_frames`` khung liên tiếp.
        ``_count_start`` ghi lại lúc bắt đầu nhả để tính thời gian chạm chính xác."""
        if pose == touch_pose:
            self._count = 0
            self._count_start = None
            return False
        if self._count == 0:
            self._count_start = now
        self._count += 1
        return self._count >= self.release_frames

    def _at_rest(self, pose: Optional[str]) -> bool:
        self._count = self._count + 1 if pose in REST_POSES else 0
        return self._count >= self.release_frames

    def update(self, raw_pose: Optional[str], confidence: float, now: float) -> List[str]:
        """Đưa vào một khung hình, trả về các sự kiện phát sinh."""
        pose = self.smoother.update(raw_pose, confidence)
        events: List[str] = []

        if self.state == "WAIT_SECOND":
            if now - self.first_tap_release > self.double_tap_window:
                events.append("SPACE")
                self._go("IDLE")
            elif pose == "PINCH_RING":
                events.append("CONFIRM")
                self._go("LOCKED")
                return events
            elif pose in TOUCH_POSES:
                events.append("SPACE")
                self._go("IDLE")
            # Nếu vừa về IDLE, rơi xuống dưới để xử lý chạm mới ngay khung này.

        if self.state == "IDLE":
            if pose == "PINCH_THUMB":
                self.touch_start = now
                self._go("THUMB_DOWN")
            elif pose == "PINCH_RING":
                self.touch_start = now
                self._go("RING_DOWN")
            elif pose == "PINCH_PINKY":
                events.append("DELETE")
                self._go("LOCKED")

        elif self.state == "THUMB_DOWN":
            if now - self.touch_start >= self.dash_time:
                events.append("DASH")
                self._go("LOCKED")
            elif self._released(pose, "PINCH_THUMB", now):
                end = self._count_start if self._count_start is not None else now
                if end - self.touch_start >= self.noise_time:
                    events.append("DOT")
                self._go("IDLE")

        elif self.state == "RING_DOWN":
            if self._released(pose, "PINCH_RING", now):
                end = self._count_start if self._count_start is not None else now
                if end - self.touch_start >= self.noise_time:
                    self.first_tap_release = end
                    self._go("WAIT_SECOND")
                else:
                    self._go("IDLE")

        elif self.state == "LOCKED":
            if self._at_rest(pose):
                self._go("IDLE")

        return events
