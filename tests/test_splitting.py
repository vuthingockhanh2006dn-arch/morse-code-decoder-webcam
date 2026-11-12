"""Kiểm thử chia tập theo nhóm (người hoặc phiên)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splitting import assign_groups_to_splits  # noqa: E402


class SplittingTest(unittest.TestCase):
    def test_every_group_in_exactly_one_split_and_none_empty(self):
        sizes = {f"user_{i}/s{i}": 100 for i in range(20)}
        assignment = assign_groups_to_splits(sizes)
        self.assertEqual(set(assignment), set(sizes))
        self.assertEqual(set(assignment.values()), {"train", "val", "test"})

    def test_ratios_are_close_to_70_15_15(self):
        sizes = {f"g{i}": 100 for i in range(20)}
        assignment = assign_groups_to_splits(sizes)
        share = {n: sum(sizes[g] for g, s in assignment.items() if s == n) / 2000 for n in ("train", "val", "test")}
        self.assertAlmostEqual(share["train"], 0.70, delta=0.06)
        self.assertAlmostEqual(share["val"], 0.15, delta=0.06)
        self.assertAlmostEqual(share["test"], 0.15, delta=0.06)

    def test_same_seed_same_result(self):
        sizes = {f"g{i}": 50 + i for i in range(12)}
        self.assertEqual(assign_groups_to_splits(sizes, seed=1), assign_groups_to_splits(sizes, seed=1))

    def test_fewer_than_three_groups_raises(self):
        with self.assertRaises(ValueError):
            assign_groups_to_splits({"a": 10, "b": 10})

    def test_three_groups_one_each(self):
        assignment = assign_groups_to_splits({"a": 10, "b": 20, "c": 30})
        self.assertEqual(sorted(assignment.values()), ["test", "train", "val"])


if __name__ == "__main__":
    unittest.main()
