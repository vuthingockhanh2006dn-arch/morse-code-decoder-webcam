"""Inventory RGB images, IPN temporal labels and landmark files separately."""
import csv
import hashlib
import json
from pathlib import Path
from config import ROOT_DIR


def audit_ipn():
    root = ROOT_DIR / "CDIOII"
    annotation = root / "IPN_HandS_annotations"
    classes = {}
    class_file = annotation / "classes.txt"
    if class_file.exists():
        with class_file.open(encoding="utf-8") as stream:
            classes = {row["id"]: {"code": row["code"], "gesture": row["gesture"], "segments": 0}
                       for row in csv.DictReader(stream)}
    annotations = sorted((annotation / "temporal_annotations").glob("*.txt"))
    for path in annotations:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3 and parts[0] in classes:
                classes[parts[0]]["segments"] += 1
    media = [p for p in root.rglob("*") if p.is_file() and "sign_language_digits" not in p.parts]
    rgb = [p for p in media if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    videos = [p for p in media if p.suffix.lower() in (".avi", ".mp4", ".mov", ".mkv")]
    return {"documentation": "https://github.com/GibranBenitez/IPN-hand/blob/master/README_HandS.md",
            "landmark_json_files": {folder: len(list((root/folder).glob("*.json")))
                for folder in ("landmarks_gt", "landmarks_mp_hands", "landmarks_mp_holistic")},
            "temporal_annotation_files": len(annotations), "classes": classes,
            "rgb_frames_present": len(rgb), "rgb_videos_present": len(videos),
            "frame_extraction": "Original 640x480 IPN Hand video -> fps=30 -> 1-based frame indices; do not use the different 320x240 frame timeline.",
            "interpretation": "Contains labeled gesture sequences. JSON frame-name keys are not RGB pixels. Click/double-click labels require visual verification before mapping to exact thumb-index/ring/pinky contact poses. Existing usable RGB data can be reused; fresh collection is only needed for missing/unmatched target poses."}


def audit():
    roots = [ROOT_DIR/"CDIOII/Morse_Webcam_Dataset/data/raw/images/sign_language_digits",
             ROOT_DIR/"datasets/data/raw/images/sign_language_digits"]
    all_hashes = set()
    sources = []
    for root in roots:
        counts, hashes = {}, set()
        for path in sorted(root.glob("*/*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes.add(digest)
            counts[path.parent.name] = counts.get(path.parent.name, 0)+1
        sources.append({"path": str(root.relative_to(ROOT_DIR)), "classes": counts,
                        "files": sum(counts.values()), "unique_hashes": len(hashes),
                        "duplicates_of_previous_source": len(hashes & all_hashes)})
        all_hashes.update(hashes)
    report = {"source": "https://github.com/ardamavi/Sign-Language-Digits-Dataset",
              "sources": sources, "unique_images_across_sources": len(all_hashes),
              "original_labels": "Digits 0-9", "morse_labeled_images_in_these_sources": 0,
              "ipn_hands": audit_ipn(),
              "use": "Digit RGB data and IPN gesture landmark/annotation data are different sources. Locate original IPN RGB videos/frames and verify target contact poses before deciding which new images are needed."}
    output = ROOT_DIR/"reports/image_dataset_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
