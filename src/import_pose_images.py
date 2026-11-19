"""Reuse actual contact poses found in Sign Language Digits: 9, 7, 6, 5.

This maps matching hand poses, not numeric commands to Morse events. It does
not fabricate participant/session IDs: filename blocks are only split proxies.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

import cv2

from config import ROOT_DIR
from image_learning import IMAGE_ROOT

SOURCE = ROOT_DIR / "CDIOII/Morse_Webcam_Dataset/data/raw/images/sign_language_digits"
POSE_MAPPING = {"9": "PINCH_THUMB", "7": "PINCH_RING", "6": "PINCH_PINKY", "5": "UNKNOWN"}


def import_images(source=SOURCE, output=IMAGE_ROOT):
    counts = {pose: 0 for pose in POSE_MAPPING.values()}
    manifest, seen = [], set()
    for original, pose in POSE_MAPPING.items():
        files = sorted((source / original).glob("*.JPG"))
        if not files:
            raise ValueError(f"Không có ảnh nguồn lớp {original}")
        for path in files:
            match = re.fullmatch(r"IMG_(\d+)\.JPG", path.name, re.IGNORECASE)
            if not match:
                raise ValueError(f"Chưa biết cách nhóm tên ảnh {path.name}")
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Không đọc được ảnh {path}")
            image = cv2.resize(image, (128, 128))
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            # Adjacent filenames tend to be correlated. This is a conservative
            # proxy grouping, NOT verified person/session metadata.
            group = f"filename_block_{int(match[1]) // 100:04d}"
            relative = Path("public_digits") / group / pose / (digest + ".png")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() and not cv2.imwrite(str(target), image):
                raise ValueError(f"Không lưu được ảnh {target}")
            counts[pose] += 1
            manifest.append([str(path.relative_to(source)), original, pose, str(relative), group,
                             hashlib.sha256(path.read_bytes()).hexdigest(), digest])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "source_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_path", "source_digit", "pose_label", "training_path", "proxy_group", "source_sha256", "pixel_sha256"])
        writer.writerows(manifest)
    protocol = {"source": "https://github.com/ardamavi/Sign-Language-Digits-Dataset",
                "mapping": POSE_MAPPING, "counts": counts, "total": sum(counts.values()),
                "mapping_reason": "9 thumb-index contact, 7 thumb-ring contact, 6 thumb-pinky contact, 5 open palm; verified on source images. Other digit classes are excluded.",
                "grouping": "filename_block_proxy", "filename_block_size": 100,
                "subject_ids_available": False,
                "limitations": "Filename blocks are not verified subjects or recording sessions. Test scores are not subject-independent performance. Open palm is only one rest pose; webcam backgrounds and other unknown gestures need extra coverage. Imported whole-image framing differs from live landmark crops; validate on webcam before claiming deployment accuracy."}
    (output / "dataset_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    return protocol


if __name__ == "__main__":
    print(json.dumps(import_images(), indent=2, ensure_ascii=False))
