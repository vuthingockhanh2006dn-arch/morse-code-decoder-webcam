"""Scale-independent fingertip contact detection, without trained models."""
from math import dist, isfinite
import time


class GestureDetector:
    PAIRS = {"PINCH_THUMB": 4, "PINCH_RING": 16, "PINCH_PINKY": 20}

    def __init__(self, touch=0.25, release=0.35, ambiguity=0.08,
                 stable_time=0.07, anchor=8):
        if anchor not in (4, 8):
            raise ValueError("anchor must be thumb (4) or index (8)")
        self.touch = touch
        self.release = release
        self.ambiguity = ambiguity
        self.stable_time = stable_time
        self.anchor = anchor
        self.pairs = dict(self.PAIRS, PINCH_THUMB=8) if anchor == 4 else dict(self.PAIRS)
        self.reset()

    def reset(self):
        self.active = None
        self.distances = {}
        self.candidate = None
        self.candidate_since = 0.

    def detect(self, landmarks, width=640, height=480, now=None):
        """Return a pose. UNKNOWN means invalid geometry or ambiguous contact.

        x and z use image-width units in MediaPipe; y must be aspect-corrected.
        Palm width and wrist-to-middle-base length provide a rotation invariant
        scale. Hysteresis retains contact until the wider release threshold.
        """
        if landmarks is None:
            self.reset()
            return None
        if len(landmarks) != 21 or width <= 0 or height <= 0:
            self.reset()
            return "UNKNOWN"
        points = [tuple(float(v) for v in p) for p in landmarks]
        if any(len(p) != 3 or not all(isfinite(v) for v in p) for p in points):
            self.reset()
            return "UNKNOWN"
        points = [(x, y * height / width, z) for x, y, z in points]
        scale = max(dist(points[0], points[9]), dist(points[5], points[17]))
        if scale < 0.015:
            self.reset()
            return "UNKNOWN"
        # MediaPipe depth is estimated; downweight it to avoid missed contacts
        # while retaining a depth separation guard for overlapping fingertips.
        weighted = [(x, y, z * .5) for x, y, z in points]
        self.distances = {label: dist(weighted[self.anchor], weighted[tip]) / scale
                          for label, tip in self.pairs.items()}
        now = time.monotonic() if now is None else now
        ranked = sorted(self.distances, key=self.distances.get)
        best, second = ranked[:2]
        if (self.distances[best] < self.touch and
                self.distances[second] - self.distances[best] < self.ambiguity):
            self.active = None
            self.candidate = None
            return "UNKNOWN"
        if self.active:
            if self.distances[self.active] < self.release:
                # Never jump from one finger to another during a contact.
                if best != self.active and self.distances[best] < self.touch:
                    self.active = self.candidate = None
                    return "UNKNOWN"
                return self.active
            self.active = self.candidate = None
            return "IDLE"
        if self.distances[best] >= self.touch:
            self.candidate = None
            return "IDLE"
        if self.candidate != best:
            self.candidate, self.candidate_since = best, now
        if now - self.candidate_since >= self.stable_time:
            self.active = best
        return self.active or "IDLE"
