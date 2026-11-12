"""Chia nhóm (người hoặc phiên quay) vào train/val/test. Không phụ thuộc pandas.

Mỗi nhóm nằm trọn trong đúng một tập, nên khung hình gần giống nhau của cùng
một phiên không thể xuất hiện ở cả train lẫn test.
"""
from __future__ import annotations

import random
from typing import Dict, Sequence

SPLIT_NAMES = ("train", "val", "test")


def assign_groups_to_splits(
    group_sizes: Dict[str, int],
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, str]:
    """Trả về ``{nhóm: tên tập}``.

    Ba nhóm lớn nhất được đặt vào train, val, test để mỗi tập chắc chắn không
    rỗng. Các nhóm còn lại lần lượt vào tập đang thiếu nhiều mẫu nhất so với
    tỷ lệ mục tiêu.
    """
    if len(group_sizes) < 3:
        raise ValueError(f"Cần ít nhất 3 nhóm để chia 3 tập, hiện có {len(group_sizes)}")

    rng = random.Random(seed)
    groups = sorted(group_sizes)
    rng.shuffle(groups)                                   # phá thế hòa khi kích thước bằng nhau
    groups.sort(key=lambda g: group_sizes[g], reverse=True)

    total = sum(group_sizes.values())
    targets = {name: ratio * total for name, ratio in zip(SPLIT_NAMES, ratios)}
    counts = {name: 0 for name in SPLIT_NAMES}
    assignment: Dict[str, str] = {}

    for name, group in zip(SPLIT_NAMES, groups[:3]):
        assignment[group] = name
        counts[name] += group_sizes[group]
    for group in groups[3:]:
        name = max(SPLIT_NAMES, key=lambda n: targets[n] - counts[n])
        assignment[group] = name
        counts[name] += group_sizes[group]
    return assignment
