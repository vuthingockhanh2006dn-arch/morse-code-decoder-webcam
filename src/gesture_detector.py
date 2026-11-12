"""Scale-independent fingertip contact detection, without trained models."""
from math import dist, isfinite


class GestureDetector:
    PAIRS = {"PINCH_THUMB": 4, "PINCH_RING": 16, "PINCH_PINKY": 20}

    def __init__(self, touch=0.25, release=0.35, ambiguity=0.07):
        self.touch = touch
        self.release = release
        self.ambiguity = ambiguity
        self.reset()

    def reset(self):
        self.active = None
        self.distances = {}

    def detect(self, landmarks, width=640, height=480):
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
        self.distances = {label: dist(points[8], points[tip]) / scale
                          for label, tip in self.PAIRS.items()}
        ranked = sorted(self.distances, key=self.distances.get)
        best, second = ranked[:2]
        if (self.distances[best] < self.touch and
                self.distances[second] - self.distances[best] < self.ambiguity):
            self.active = None
            return "UNKNOWN"
        if self.active and self.distances[self.active] < self.release:
            return self.active
        self.active = best if self.distances[best] < self.touch else None
        return self.active or "IDLE"
