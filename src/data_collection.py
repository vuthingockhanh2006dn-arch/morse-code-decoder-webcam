"""Webcam tool to record labelled hand-landmark samples.

Only the 21 landmarks (x, y, z) are stored, not the video, so no images of the
participant are kept. Each run is one *session* and is saved to
``data/raw/<task>/<subject>/<session>.csv``. Keep different people, lighting,
backgrounds and distances in separate sessions so the split can be done by
session (see preprocessing.py).

Usage::

    python src/data_collection.py --task morse --subject user_01 --session s01_daylight
    python src/data_collection.py --task sign  --subject user_01 --session s01 --labels NOTHING,A,B,C

Keys:
    0-9     start recording the label with that index (shown on screen)
    SPACE   pause recording
    u       undo: remove the samples of the last recording burst
    q / ESC save and quit
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_SIGN_LABELS, FEATURE_COLUMNS, META_COLUMNS, MORSE_POSE_LABELS, RAW_DIR  # noqa: E402
from hand_tracker import HandTracker  # noqa: E402

MORSE_HINTS = {
    "IDLE": "open hand / rest pose, no fingers touching",
    "PINCH_THUMB": "index tip touches thumb tip (used for DOT and DASH)",
    "PINCH_RING": "index tip touches ring-finger tip (used for SPACE and CONFIRM)",
    "PINCH_PINKY": "index tip touches pinky tip (used for DELETE)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["morse", "sign"], default="morse")
    parser.add_argument("--subject", required=True, help="Anonymous participant id, e.g. user_01")
    parser.add_argument("--session", required=True, help="Session id, e.g. s01_daylight_50cm")
    parser.add_argument("--labels", help="Comma-separated labels (sign task only)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--every", type=int, default=2, help="Save every N-th frame to reduce near-duplicates")
    args = parser.parse_args()

    if args.task == "morse":
        labels = MORSE_POSE_LABELS
    else:
        labels = args.labels.split(",") if args.labels else DEFAULT_SIGN_LABELS
    if len(labels) > 10:
        raise SystemExit("At most 10 labels per session (keys 0-9).")

    out_path = RAW_DIR / args.task / args.subject / f"{args.session}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, bursts = [], []           # bursts: start index of each recording burst
    counts = {lab: 0 for lab in labels}

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    tracker = HandTracker()
    current, frame_idx = None, 0

    print(__doc__)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hand = tracker.process(frame)
            frame_idx += 1

            if hand is not None:
                tracker.draw(frame, hand)
                if current is not None and frame_idx % args.every == 0:
                    rows.append([args.subject, args.session, current, hand.handedness, time.time(),
                                 *hand.landmarks.reshape(-1).tolist()])
                    counts[current] += 1

            # ---- overlay -------------------------------------------------- #
            status = f"REC: {current}" if current else "PAUSED"
            color = (0, 0, 255) if current else (200, 200, 200)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if current and hand is None:
                cv2.putText(frame, "No hand detected - not saving", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for i, lab in enumerate(labels):
                text = f"{i}: {lab} ({counts[lab]})"
                cv2.putText(frame, text, (10, 95 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            if current and args.task == "morse":
                cv2.putText(frame, MORSE_HINTS[current], (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow("Data collection", frame)

            # ---- keys ----------------------------------------------------- #
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                current = None
            elif key == ord("u") and bursts:
                start = bursts.pop()
                for r in rows[start:]:
                    counts[r[2]] -= 1
                del rows[start:]
                current = None
            elif ord("0") <= key <= ord("9") and key - ord("0") < len(labels):
                current = labels[key - ord("0")]
                bursts.append(len(rows))
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    if not rows:
        print("Nothing recorded.")
        return
    new_file = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(META_COLUMNS + FEATURE_COLUMNS)
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples to {out_path}: {counts}")


if __name__ == "__main__":
    main()
