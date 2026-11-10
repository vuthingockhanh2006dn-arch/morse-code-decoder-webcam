"""Shared paths, label definitions and runtime thresholds.

Runtime thresholds live in ``models/scaler_config.json`` so they can be tuned on
the validation set without touching code. Values missing from that file fall
back to ``DEFAULT_CONFIG`` below.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIG_PATH = MODELS_DIR / "scaler_config.json"

# --------------------------------------------------------------------------- #
# Tasks & labels
# --------------------------------------------------------------------------- #
# The Morse classifier recognises static *hand poses* from a single frame.
# A single frame cannot tell a quick tap (DOT) from a long hold (DASH), or one
# tap (SPACE) from two taps (CONFIRM). Those are decided over time by the FSM
# in ``fsm_controller.py``:
#
#   PINCH_THUMB  (index + thumb)  -> short tap = DOT,   long hold = DASH
#   PINCH_RING   (index + ring)   -> 1 tap     = SPACE, 2 taps    = CONFIRM
#   PINCH_PINKY  (index + pinky)  -> 1 tap     = DELETE
#   IDLE         (open hand / rest pose)
MORSE_POSE_LABELS = ["IDLE", "PINCH_THUMB", "PINCH_RING", "PINCH_PINKY"]

# Command events produced by the FSM (the 5 labels from the README).
EVENTS = ["DOT", "DASH", "CONFIRM", "SPACE", "DELETE"]

# Default labels for the static-sign mode. Change them with
# ``data_collection.py --task sign --labels A,B,C,...``.
DEFAULT_SIGN_LABELS = ["NOTHING", "A", "B", "C", "D", "E"]

TASKS = ("morse", "sign")

# MediaPipe landmark indices used by the rule-based detector.
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 9, 12, 16, 20

N_LANDMARKS = 21
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(N_LANDMARKS) for axis in ("x", "y", "z")]
META_COLUMNS = ["subject", "session", "label", "handedness", "timestamp"]

ALGORITHMS = ("knn", "svm", "rf")


def model_path(task: str, algo: str) -> Path:
    """File name of a trained model, e.g. ``models/gesture_svm_model.pkl``."""
    prefix = "gesture" if task == "morse" else "sign"
    return MODELS_DIR / f"{prefix}_{algo}_model.pkl"


# --------------------------------------------------------------------------- #
# Runtime configuration
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    # Rule-based touch detection (normalised Euclidean distance).
    "touch_threshold": 0.25,      # d_hat < touch_threshold  -> touching
    "release_threshold": 0.35,    # d_hat > release_threshold -> released (hysteresis)
    # Timing (seconds).
    "noise_time": 0.05,           # touches shorter than this are ignored
    "dash_time": 0.40,            # hold >= dash_time -> DASH
    "double_tap_window": 0.45,    # second ring tap must start within this window -> CONFIRM
    # Stabilisation.
    "confidence_threshold": 0.75, # ML predictions below this become UNKNOWN
    "smoothing_window": 5,        # frames used for majority voting
    "release_frames": 3,          # consecutive non-touch frames needed to count a release
    # Static-sign mode.
    "sign_stable_frames": 10,     # frames a sign must stay stable before it is committed
    # Filled in by train_models.py.
    "best_model": {"morse": None, "sign": None},
}


def load_config() -> Dict[str, Any]:
    """Return DEFAULT_CONFIG overridden by values in ``models/scaler_config.json``."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as f:
            user_cfg = json.load(f)
        for key, value in user_cfg.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
