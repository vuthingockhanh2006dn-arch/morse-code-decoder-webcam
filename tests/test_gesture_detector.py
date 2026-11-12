"""Geometry regression tests; no webcam, numpy or MediaPipe required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gesture_detector import GestureDetector


def hand(tip=4, gap=.02):
    points = [[.5, .5, 0.] for _ in range(21)]
    points[0] = [.5, .8, 0.]
    points[9] = [.5, .5, 0.]
    points[5] = [.35, .5, 0.]
    points[17] = [.65, .5, 0.]
    points[8] = [.4, .3, 0.]
    points[4] = [.1, .3, 0.]
    points[16] = [.65, .3, 0.]
    points[20] = [.8, .3, 0.]
    points[tip] = [.4 + gap, .3, 0.]
    return points


class GestureDetectorTest(unittest.TestCase):
    def test_each_contact(self):
        for label, tip in GestureDetector.PAIRS.items():
            with self.subTest(label=label):
                self.assertEqual(GestureDetector().detect(hand(tip)), label)

    def test_scale_translation_and_mirror_invariance(self):
        original = hand(20)
        for scale in (.6, 1., 1.5):
            points = [[.9 - x * scale, y * scale + .03, z * scale] for x, y, z in original]
            self.assertEqual(GestureDetector().detect(points), "PINCH_PINKY")

    def test_hysteresis_and_release(self):
        detector = GestureDetector()
        self.assertEqual(detector.detect(hand(gap=.06)), "PINCH_THUMB")
        self.assertEqual(detector.detect(hand(gap=.09)), "PINCH_THUMB")
        self.assertEqual(detector.detect(hand(gap=.12)), "IDLE")
        self.assertEqual(detector.detect(hand(gap=.09)), "IDLE")

    def test_ambiguous_ring_pinky_rejected(self):
        points = hand(16)
        points[20] = [.425, .3, 0.]
        self.assertEqual(GestureDetector().detect(points), "UNKNOWN")

    def test_missing_and_invalid_hand_reset_contact(self):
        detector = GestureDetector()
        detector.detect(hand())
        self.assertIsNone(detector.detect(None))
        self.assertIsNone(detector.active)
        self.assertEqual(detector.detect([[0., 0., 0.]] * 21), "UNKNOWN")
        self.assertEqual(detector.detect([[float('nan'), 0., 0.]] * 21), "UNKNOWN")

    def test_aspect_ratio_correction(self):
        points = hand(16)
        tall = [[x, y / 2, z] for x, y, z in points]
        self.assertEqual(GestureDetector().detect(tall, 640, 960),
                         GestureDetector().detect(points, 640, 480))


if __name__ == "__main__":
    unittest.main()
