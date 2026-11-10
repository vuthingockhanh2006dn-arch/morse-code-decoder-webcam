"""Finite State Machine that turns per-frame hand poses into Morse commands.

Input per frame: a pose label (``IDLE``, ``PINCH_THUMB``, ``PINCH_RING``,
``PINCH_PINKY``, ``UNKNOWN`` or ``None`` when no hand is visible), its
confidence, and a timestamp in seconds.

Output: at most one command event per gesture (``DOT``, ``DASH``, ``CONFIRM``,
``SPACE``, ``DELETE``).

Anti-repeat rules (debouncing):
  * Poses are smoothed with a majority vote over the last N frames.
  * A touch only counts as released after ``release_frames`` consecutive
    frames without that touch.
  * After DASH, CONFIRM or DELETE the machine is LOCKED until the hand returns
    to the rest pose (IDLE) or leaves the frame, so holding a gesture in front
    of a 30 FPS camera never produces repeated input.

State diagram::

    IDLE --PINCH_THUMB--> THUMB_DOWN --release (<dash_time)--> IDLE        emit DOT
                                     --held >= dash_time-----> LOCKED      emit DASH
    IDLE --PINCH_RING---> RING_DOWN  --release--------------> WAIT_SECOND
    WAIT_SECOND --PINCH_RING within window-------------------> LOCKED      emit CONFIRM
    WAIT_SECOND --window expires / other pose----------------> IDLE        emit SPACE
    IDLE --PINCH_PINKY--------------------------------------->  LOCKED     emit DELETE
    LOCKED --IDLE or no hand for release_frames--------------> IDLE
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional

IDLE_POSES = {"IDLE", None}
TOUCH_POSES = {"PINCH_THUMB", "PINCH_RING", "PINCH_PINKY"}


class PoseSmoother:
    """Confidence threshold + majority vote over a sliding window."""

    def __init__(self, window: int = 5, confidence_threshold: float = 0.75) -> None:
        self.window: Deque[Optional[str]] = deque(maxlen=max(1, window))
        self.confidence_threshold = confidence_threshold

    def update(self, pose: Optional[str], confidence: float = 1.0) -> Optional[str]:
        if pose is not None and confidence < self.confidence_threshold:
            pose = "UNKNOWN"
        self.window.append(pose)
        winner, count = Counter(self.window).most_common(1)[0]
        # Require a strict majority; otherwise report UNKNOWN (transition frames).
        return winner if count > len(self.window) // 2 else "UNKNOWN"

    def reset(self) -> None:
        self.window.clear()


class MorseFSM:
    """Converts smoothed poses into Morse command events."""

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
        self._release_count = 0
        self._release_time: Optional[float] = None
        self.hold_progress = 0.0  # 0..1, how close the current thumb hold is to a DASH
        self.smoother.reset()

    # ------------------------------------------------------------------ #
    def _released(self, pose: Optional[str], touch_pose: str, now: float) -> bool:
        """True once ``touch_pose`` has been absent for ``release_frames`` frames."""
        if pose == touch_pose:
            self._release_count = 0
            self._release_time = None
            return False
        if self._release_count == 0:
            self._release_time = now
        self._release_count += 1
        return self._release_count >= self.release_frames

    def _rest(self, pose: Optional[str]) -> bool:
        """True once the hand has been at rest for ``release_frames`` frames."""
        if pose in IDLE_POSES:
            self._release_count += 1
        else:
            self._release_count = 0
        return self._release_count >= self.release_frames

    def _go(self, state: str) -> None:
        self.state = state
        self._release_count = 0
        self._release_time = None

    # ------------------------------------------------------------------ #
    def update(self, raw_pose: Optional[str], confidence: float, now: float) -> List[str]:
        """Feed one frame; return the list of events emitted (usually empty)."""
        pose = self.smoother.update(raw_pose, confidence)
        events: List[str] = []
        self.hold_progress = 0.0

        if self.state == "WAIT_SECOND":
            if pose == "PINCH_RING":
                events.append("CONFIRM")
                self._go("LOCKED")
                return events
            if now - self.first_tap_release > self.double_tap_window or pose in TOUCH_POSES:
                events.append("SPACE")
                self._go("IDLE")
                # fall through so a new touch in this frame is handled below

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
            held = now - self.touch_start
            self.hold_progress = min(1.0, held / self.dash_time)
            if held >= self.dash_time:
                events.append("DASH")
                self._go("LOCKED")
            elif self._released(pose, "PINCH_THUMB", now):
                release_time = self._release_time if self._release_time is not None else now
                if release_time - self.touch_start >= self.noise_time:
                    events.append("DOT")
                self._go("IDLE")

        elif self.state == "RING_DOWN":
            if self._released(pose, "PINCH_RING", now):
                release_time = self._release_time if self._release_time is not None else now
                if release_time - self.touch_start >= self.noise_time:
                    self.first_tap_release = release_time
                    self._go("WAIT_SECOND")
                else:
                    self._go("IDLE")

        elif self.state == "LOCKED":
            if self._rest(pose):
                self._go("IDLE")

        return events

    @property
    def last_smoothed(self) -> Optional[str]:
        return self.smoother.window[-1] if self.smoother.window else None


class SignCommitter:
    """Debouncing for the static-sign mode.

    A sign is committed once it has been the smoothed prediction for
    ``stable_frames`` frames. The same sign is not committed again until the
    hand leaves the frame or shows the rest label (e.g. ``NOTHING``) or a
    different sign.
    """

    def __init__(self, cfg: Dict[str, Any], rest_label: str = "NOTHING") -> None:
        self.stable_frames = int(cfg["sign_stable_frames"])
        self.rest_label = rest_label
        self.smoother = PoseSmoother(int(cfg["smoothing_window"]), float(cfg["confidence_threshold"]))
        self.reset()

    def reset(self) -> None:
        self.candidate: Optional[str] = None
        self.count = 0
        self.locked_label: Optional[str] = None
        self.smoother.reset()

    def update(self, raw_label: Optional[str], confidence: float) -> Optional[str]:
        label = self.smoother.update(raw_label, confidence)
        if label in (None, self.rest_label):
            self.locked_label = None
            self.candidate, self.count = None, 0
            return None
        if label == "UNKNOWN":
            return None
        if label != self.candidate:
            self.candidate, self.count = label, 0
        self.count += 1
        if self.count >= self.stable_frames and label != self.locked_label:
            self.locked_label = label
            return label
        return None

    @property
    def progress(self) -> float:
        if self.candidate is None or self.candidate == self.locked_label:
            return 0.0
        return min(1.0, self.count / self.stable_frames)
