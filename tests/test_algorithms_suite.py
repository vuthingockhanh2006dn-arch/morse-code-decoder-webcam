"""Kiểm thử việc chạy script test_algorithms."""
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from test_algorithms import test_feature_engineering, test_fsm_and_morse, test_ml_algorithms


class TestAlgorithmsSuite(unittest.TestCase):
    def test_feature_engineering_suite(self):
        self.assertTrue(test_feature_engineering())

    def test_ml_algorithms_suite(self):
        self.assertTrue(test_ml_algorithms())

    def test_fsm_and_morse_suite(self):
        self.assertTrue(test_fsm_and_morse())


if __name__ == "__main__":
    unittest.main()
