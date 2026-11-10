# Dataset Description / Mô tả bộ dữ liệu

> Fill in the `TODO` fields with your real numbers after collecting data.
> `python src/preprocessing.py --task <task>` writes the statistics to
> `data/processed/<task>/dataset_stats.json` and the plot to
> `reports/figures/<task>_class_distribution.png`.

## 1. Source / Nguồn dữ liệu
- **Self-collected** with `src/data_collection.py` using a laptop webcam. No external dataset is used.
- Only the 21 MediaPipe hand landmarks (x, y, z) are stored. **No images or video** of participants are saved.
- Participants are identified by anonymous ids (`user_01`, `user_02`, ...) and gave consent to take part.

## 2. Label meaning / Ý nghĩa nhãn

### 2.1 Morse task (`data/raw/morse/`)
These are **application control gestures defined by the project**. They are not part of any sign language.

A single frame only shows the hand **pose**. It cannot tell a quick tap from a long hold, or one tap from two taps. So the classifier learns **4 pose classes**, and the 5 command labels in the README come from the time-based FSM (`src/fsm_controller.py`):

| Pose class (model label) | Hand pose | Command produced by the FSM |
|---|---|---|
| `IDLE` | Open hand, no fingertips touching (rest pose) | – (required between gestures) |
| `PINCH_THUMB` | Index tip touches thumb tip | quick tap → `DOT` (`.`), hold ≥ `dash_time` → `DASH` (`-`) |
| `PINCH_RING` | Index tip touches ring-finger tip | 1 tap → `SPACE`, 2 taps within `double_tap_window` → `CONFIRM` |
| `PINCH_PINKY` | Index tip touches pinky tip | 1 tap → `DELETE` |

### 2.2 Static-sign task (`data/raw/sign/`)
- Labels are chosen with `--labels` (default: `NOTHING,A,B,C,D,E`). TODO: state which sign alphabet they come from (e.g. ASL fingerspelling letters that are static and one-handed).
- `NOTHING` is the rest/background class used for debouncing.
- Recognising isolated static signs is **not** sign-language translation.

## 3. Collection protocol / Quy trình thu thập
1. One run of `data_collection.py` = one **session** → `data/raw/<task>/<subject>/<session>.csv`.
2. Use a session name that describes the conditions, e.g. `s01_daylight_50cm`, `s02_lamp_80cm_cluttered`.
3. For each session: press the label key, perform the pose with small variations (rotation, tilt, distance), then press SPACE to pause. Press `u` to undo a bad burst.
4. Every 2nd frame is saved (`--every 2`) to reduce near-duplicate frames.
5. Vary the conditions: lighting (daylight / lamp / dim), background, distance (≈30–80 cm), left and right hand.

**CSV columns:** `subject, session, label, handedness, timestamp, x0, y0, z0, ..., x20, y20, z20`

## 4. Cleaning & preprocessing / Làm sạch và tiền xử lý
- Drop rows with missing values or no hand detected. Rows are only recorded when a hand is detected.
- Drop exact duplicate rows and degenerate hands.
- Normalise: wrist-centred → mirror left hands → scale by hand size → 63-value vector.

## 5. Split / Chia tập
- 70 / 15 / 15 **by session** (`subject/session`): one session never appears in two subsets.
- At least 3 sessions are required. `--allow-frame-split` exists for pilot runs only.

## 6. Statistics / Thống kê (TODO)

| Item | Morse | Sign |
|---|---|---|
| Classes | 4 | TODO |
| Samples (after cleaning) | TODO | TODO |
| Subjects | TODO | TODO |
| Sessions | TODO | TODO |
| Max/min class ratio | TODO | TODO |

## 7. Known issues / Quan sát chất lượng (TODO, based on your experiments)
- Which classes look alike (check the confusion matrix): e.g. `PINCH_RING` vs `PINCH_PINKY` from oblique angles.
- Frames lost to motion blur or self-occlusion.
- Differences between collection conditions and real use.
