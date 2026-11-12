"""Kiểm thử chuẩn hóa landmark và đặc trưng."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DISTANCE_COLUMNS, FEATURE_COLUMNS  # noqa: E402
from feature_engineering import (  # noqa: E402
    compute_additional_distances,
    extract_full_feature_vector,
    is_fully_visible,
    normalize_landmarks,
)


def make_hand(scale=1.0, dx=0.0):
    pts = [[0.2 + i * 0.03, 0.2 + (i % 5) * 0.1, 0.01 * (i % 3)] for i in range(21)]
    pts[0] = [0.5, 0.8, 0.0]
    return [[x * scale + dx, y * scale, z * scale] for x, y, z in pts]


class FeatureEngineeringTest(unittest.TestCase):
    def test_wrist_is_origin(self):
        v = normalize_landmarks(make_hand())
        self.assertEqual(len(v), 63)
        self.assertEqual(v[:3], [0.0, 0.0, 0.0])

    def test_left_hand_is_mirrored(self):
        right = normalize_landmarks(make_hand(), "Right")
        left = normalize_landmarks(make_hand(), "Left")
        self.assertAlmostEqual(left[3], -right[3], places=6)
        self.assertAlmostEqual(left[4], right[4], places=6)

    def test_invariant_to_position_and_size(self):
        a = extract_full_feature_vector(make_hand())
        b = extract_full_feature_vector(make_hand(scale=0.5, dx=0.1))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=5)

    def test_feature_vector_matches_columns(self):
        v = extract_full_feature_vector(make_hand())
        self.assertEqual(len(v), 66)
        self.assertEqual(len(v), len(FEATURE_COLUMNS))
        self.assertEqual(FEATURE_COLUMNS[-3:], DISTANCE_COLUMNS)

    def test_touching_fingers_give_zero_distance(self):
        pts = make_hand()
        pts[4] = list(pts[8])  # đầu ngón cái trùng đầu ngón trỏ
        self.assertAlmostEqual(compute_additional_distances(pts)["dist_thumb_index"], 0.0, places=6)

    def test_degenerate_hand_raises(self):
        with self.assertRaises(ValueError):
            normalize_landmarks([[0.5, 0.5, 0.0]] * 21)

    def test_visibility_check(self):
        self.assertTrue(is_fully_visible(make_hand()))
        pts = make_hand()
        pts[8][0] = 1.3
        self.assertFalse(is_fully_visible(pts))

    def test_flat_input_is_accepted(self):
        flat = [v for p in make_hand() for v in p]
        self.assertEqual(len(extract_full_feature_vector(flat)), 66)


if __name__ == "__main__":
    unittest.main()
