"""Thin wrapper around MediaPipe Hands (legacy ``mp.solutions`` API)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

_mp_hands = mp.solutions.hands
_mp_draw = mp.solutions.drawing_utils
_mp_styles = mp.solutions.drawing_styles


@dataclass
class HandResult:
    landmarks: np.ndarray       # (21, 3) normalised image coordinates from MediaPipe
    handedness: str             # "Left" or "Right"
    score: float                # handedness confidence
    raw: object                 # original MediaPipe landmark list (for drawing)


class HandTracker:
    """Detect a single hand per frame.

    The frame passed to :meth:`process` must already be mirrored (selfie view),
    which is what MediaPipe's handedness labels assume.
    """

    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5) -> None:
        self.hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        lm = result.multi_hand_landmarks[0]
        cls = result.multi_handedness[0].classification[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        return HandResult(landmarks=pts, handedness=cls.label, score=float(cls.score), raw=lm)

    @staticmethod
    def draw(frame_bgr: np.ndarray, hand: HandResult) -> None:
        _mp_draw.draw_landmarks(
            frame_bgr, hand.raw, _mp_hands.HAND_CONNECTIONS,
            _mp_styles.get_default_hand_landmarks_style(),
            _mp_styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        self.hands.close()
