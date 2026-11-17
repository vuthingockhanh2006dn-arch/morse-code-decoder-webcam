# Morse Webcam — real-data collection package

**Important: This ZIP is a data-collection package, not a preloaded 2,000-sample dataset.** The ZIP contains **zero downloaded photos/videos** because the creation environment could not reach the public dataset host. No dataset counts are claimed until verified locally.

## Step 1: Download >2,000 genuine public images

Requires Python 3.9+ and Internet. From this folder:

```bash
python scripts/download_dataset.py --min-images 2000 --max-images 2200
```

The script fetches the public Sign Language Digits dataset from GitHub, verifies image file signatures, computes SHA-256, and writes `metadata/image_manifest.csv` and `reports/download_report.json`. If fewer than 2,000 valid files are available, it exits with an error. Re-running is supported.

**Critical limitation:** Source images are labeled digits 0–9. They are **not** automatically labeled DOT, DASH, SPACE, CONFIRM or DELETE. They are suitable for pipeline testing and hand-pose pretraining, not a validated Morse-class classifier. Check the upstream repository's license before redistribution.

Source: https://github.com/ardamavi/Sign-Language-Digits-Dataset

## Step 2: Record genuine Morse gesture videos

```bash
pip install opencv-python
python scripts/record_morse_video.py --person P01 --seconds 3
```

Press 1–6 to record a labeled gesture video, `q` to quit. Repeat for at least three people and varied lighting, distance, and backgrounds. For DOT/DASH and SPACE/CONFIRM, include the complete gesture event, not just a static pose. Use separate sessions for test/validation.

## Step 3: Other real video sources

- IPN Hand (RGB video): https://gibranbenitez.github.io/IPN_Hand/
- SCUT-DHGA: https://github.com/SCUT-BIP-Lab/SCUT-DHGA
- Jester: https://www.qualcomm.com/developer/software/jester-dataset

Download from their official providers under applicable terms. These are **not** automatically Morse-labeled, and no video files are bundled.

## Directory structure

- `data/raw/images/sign_language_digits/`: downloaded real images by original class
- `data/raw/videos/<person>/<session>/`: locally recorded genuine videos
- `data/processed/frames/`: reserved for extracted frames
- `data/processed/landmarks/`: reserved for MediaPipe features
- `metadata/image_manifest.csv`: verified image provenance and hashes
- `metadata/video_manifest.csv`: labeled webcam recordings
- `reports/download_report.json`: actual count verification

## Research protocol

Use the project's six labels: DOT, DASH, SPACE, CONFIRM, DELETE, UNKNOWN. A single frame cannot distinguish DOT from DASH or SPACE from CONFIRM; these are temporal actions. Split train/validation/test by person or session, not randomly by adjacent frames. Target 70/15/15 and report macro-F1. Do not present the digit labels as Morse ground truth.
