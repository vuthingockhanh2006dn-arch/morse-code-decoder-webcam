"""Feature extraction, cleaning, statistics and session-wise data splitting.

Feature pipeline (21 landmarks -> 63 features):
  1. Translate so the wrist (landmark 0) is the origin.
  2. Mirror the x axis for left hands so left/right hands share one space.
  3. Scale by the largest landmark distance from the wrist (distance invariance).
  4. Flatten to a 63-value vector [x0, y0, z0, ..., x20, y20, z20].

Run as a script to turn ``data/raw/<task>/**/*.csv`` into
``data/processed/<task>/{train,val,test}.csv``::

    python src/preprocessing.py --task morse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    FEATURE_COLUMNS, FIGURES_DIR, INDEX_TIP, MIDDLE_MCP, PINKY_TIP, PROCESSED_DIR,
    RAW_DIR, RING_TIP, THUMB_TIP, WRIST,
)

EPS = 1e-6


# --------------------------------------------------------------------------- #
# Feature extraction (used by training *and* the live app)
# --------------------------------------------------------------------------- #
def normalize_landmarks(landmarks: np.ndarray, handedness: Optional[str] = "Right") -> np.ndarray:
    """Normalise a (21, 3) landmark array and return a flat 63-value vector."""
    pts = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[WRIST]
    if handedness == "Left":
        pts[:, 0] = -pts[:, 0]
    scale = float(np.max(np.linalg.norm(pts, axis=1)))
    if scale < EPS:
        raise ValueError("Degenerate hand: all landmarks collapse to one point")
    return (pts / scale).reshape(-1)


def normalized_distance(landmarks: np.ndarray, i: int, j: int) -> float:
    """2D Euclidean distance between landmarks i and j divided by palm size
    (wrist -> middle-finger MCP), so it does not depend on camera distance."""
    pts = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
    ref = np.hypot(pts[MIDDLE_MCP, 0] - pts[WRIST, 0], pts[MIDDLE_MCP, 1] - pts[WRIST, 1])
    return float(d / (ref + EPS))


def pinch_distances(landmarks: np.ndarray) -> Dict[str, float]:
    """Distances for the three control pinches used in Morse mode."""
    return {
        "PINCH_THUMB": normalized_distance(landmarks, INDEX_TIP, THUMB_TIP),
        "PINCH_RING": normalized_distance(landmarks, INDEX_TIP, RING_TIP),
        "PINCH_PINKY": normalized_distance(landmarks, INDEX_TIP, PINKY_TIP),
    }


# --------------------------------------------------------------------------- #
# Dataset building
# --------------------------------------------------------------------------- #
def load_raw(task: str):
    import pandas as pd

    files = sorted((RAW_DIR / task).rglob("*.csv"))
    if not files:
        raise SystemExit(f"No raw CSV files in {RAW_DIR / task}. Run src/data_collection.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} raw rows from {len(files)} file(s).")
    return df


def clean(df):
    """Drop broken rows, exact duplicates and degenerate hands; normalise features."""
    import pandas as pd

    report = {"raw_rows": int(len(df))}
    df = df.dropna(subset=FEATURE_COLUMNS + ["label", "session", "subject"])
    report["after_dropna"] = int(len(df))
    df = df.drop_duplicates(subset=FEATURE_COLUMNS + ["label"])
    report["after_dedup"] = int(len(df))

    feats, keep = [], []
    for idx, row in df.iterrows():
        try:
            feats.append(normalize_landmarks(row[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
                                             row.get("handedness", "Right")))
            keep.append(idx)
        except ValueError:
            continue
    out = df.loc[keep, ["subject", "session", "label"]].reset_index(drop=True)
    out = pd.concat([out, pd.DataFrame(np.vstack(feats), columns=FEATURE_COLUMNS)], axis=1)
    report["after_clean"] = int(len(out))
    return out, report


def split_by_session(df, ratios: Sequence[float] = (0.70, 0.15, 0.15), seed: int = 42,
                     allow_frame_split: bool = False):
    """Split so that every recording session lands entirely in one subset.

    Sessions are identified by ``subject/session``. Sessions are shuffled and
    assigned greedily until each subset reaches its target share of frames.
    """
    groups = (df["subject"].astype(str) + "/" + df["session"].astype(str)).to_numpy()
    unique = np.unique(groups)

    if len(unique) < 3:
        if not allow_frame_split:
            raise SystemExit(
                f"Only {len(unique)} session(s) found. A session-wise split needs at least 3 "
                "(record more sessions/people), or pass --allow-frame-split for a pilot run "
                "(test scores will then be optimistic because of near-duplicate frames)."
            )
        print("WARNING: frame-level random split. Test scores will be over-optimistic.")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n_tr, n_va = int(ratios[0] * len(df)), int(ratios[1] * len(df))
        parts = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
        return tuple(df.iloc[p].reset_index(drop=True) for p in parts)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    sizes = {g: int((groups == g).sum()) for g in unique}
    total = sum(sizes.values())
    assign = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    targets = dict(zip(assign, ratios))
    # Make sure val and test get at least one session each.
    assign["val"].append(unique[0]); counts["val"] += sizes[unique[0]]
    assign["test"].append(unique[1]); counts["test"] += sizes[unique[1]]
    for g in unique[2:]:
        # Give the session to the subset furthest below its target share.
        name = min(assign, key=lambda k: counts[k] / total - targets[k])
        assign[name].append(g); counts[name] += sizes[g]

    out = []
    for name in ("train", "val", "test"):
        mask = np.isin(groups, assign[name])
        out.append(df[mask].reset_index(drop=True))
        print(f"{name:>5}: {mask.sum():6d} frames, sessions = {sorted(assign[name])}")
    return tuple(out)


def save_statistics(task: str, df, splits, report: dict) -> None:
    """Write dataset statistics (for Chapter 3 of the report) and a class-distribution plot."""
    import pandas as pd

    stats = {
        "task": task,
        "cleaning": report,
        "n_classes": int(df["label"].nunique()),
        "n_samples": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_sessions": int((df["subject"].astype(str) + "/" + df["session"].astype(str)).nunique()),
        "class_counts": {k: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "split_counts": {name: {k: int(v) for k, v in s["label"].value_counts().sort_index().items()}
                         for name, s in zip(("train", "val", "test"), splits)},
    }
    counts = pd.Series(stats["class_counts"])
    stats["imbalance_ratio_max_over_min"] = float(counts.max() / max(counts.min(), 1))

    out_dir = PROCESSED_DIR / task
    with (out_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(stats["split_counts"]).fillna(0).astype(int)
        ax = table.plot(kind="bar", figsize=(8, 4), rot=0)
        ax.set_title(f"Class distribution per split ({task})")
        ax.set_ylabel("Samples")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{task}_class_distribution.png", dpi=150)
        plt.close()
    except ImportError:
        print("matplotlib not installed: skipping class distribution plot.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["morse", "sign"], default="morse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-frame-split", action="store_true",
                        help="Allow a random frame split when fewer than 3 sessions exist (pilot only).")
    args = parser.parse_args()

    df, report = clean(load_raw(args.task))
    splits = split_by_session(df, seed=args.seed, allow_frame_split=args.allow_frame_split)

    out_dir = PROCESSED_DIR / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in zip(("train", "val", "test"), splits):
        part.to_csv(out_dir / f"{name}.csv", index=False)
    save_statistics(args.task, df, splits, report)
    print(f"Saved processed data to {out_dir}")


if __name__ == "__main__":
    main()
