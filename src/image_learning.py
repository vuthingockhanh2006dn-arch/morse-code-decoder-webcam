"""Supervised pose classification from image pixels (HOG), not landmarks."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from config import ROOT_DIR, POSE_LABELS

IMAGE_ROOT = ROOT_DIR / "data" / "images" / "morse"
IMAGE_MODEL = ROOT_DIR / "models" / "image_pose_model.joblib"
IMAGE_REPORT = ROOT_DIR / "reports" / "image_learning"
LABELS = tuple(POSE_LABELS)
FEATURE_VERSION = "gray64-hog16-8-8-9-v1"


def decode_image(value):
    if not isinstance(value, str) or len(value) > 600_000:
        raise ValueError("Ảnh JPEG/PNG base64 không hợp lệ hoặc quá lớn")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("Ảnh base64 không hợp lệ") from exc
    # Inspect dimensions before OpenCV allocates a full decoded image.
    from PIL import Image
    import io
    try:
        with Image.open(io.BytesIO(raw)) as header:
            if header.format not in ("JPEG", "PNG") or not 16 <= min(header.size) or max(header.size) > 2048:
                raise ValueError("Ảnh cần có kích thước từ 16 đến 2048 pixel")
    except Exception as exc:
        raise ValueError("Không đọc được ảnh JPEG/PNG hợp lệ") from exc
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không giải mã được ảnh")
    return image


def image_features(image):
    """Identical preprocessing for collection, training and live inference."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Cần ảnh màu BGR")
    gray = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2., (8, 8)).apply(gray)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    return hog.compute(gray).reshape(-1).astype(np.float32)


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", value):
        raise ValueError("Mã người/phiên chỉ gồm chữ không dấu, số, _ hoặc - (tối đa 48 ký tự)")
    return value


def training_variants(image, rng):
    """Only applied AFTER splitting groups. Preserve finger identity on both hands."""
    variants = [image, cv2.flip(image, 1)]
    for source in variants[:2]:
        angle = float(rng.uniform(-8, 8))
        h, w = source.shape[:2]
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.)
        transformed = cv2.warpAffine(source, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        transformed = cv2.GaussianBlur(transformed, (3,3), .5)
        noise = rng.normal(0, 5, transformed.shape)
        transformed = np.clip(transformed.astype(float)*rng.uniform(.55, .85)+noise, 0, 255).astype(np.uint8)
        variants.append(transformed)
    return variants


def save_sample(encoded, label, subject, session, root=IMAGE_ROOT, train_only=False):
    if label not in LABELS:
        raise ValueError("Nhãn ảnh không hợp lệ")
    subject, session = safe_name(subject), safe_name(session)
    if not isinstance(train_only, bool):
        raise ValueError("train_only phải là true/false")
    image = cv2.resize(decode_image(encoded), (128, 128))
    # Reject exact duplicate normalized pixels across the entire dataset.
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if next(root.glob(f"*/*/*/{digest}.png"), None):
        return {"saved": False, "message": "Bỏ qua ảnh trùng hoàn toàn"}
    folder = root / subject / session / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest}.png"
    if not cv2.imwrite(str(path), image):
        raise ValueError("Không lưu được ảnh huấn luyện")
    if train_only:
        metadata = root / "calibration_groups.json"
        groups = set(json.loads(metadata.read_text(encoding="utf-8"))) if metadata.exists() else set()
        groups.add(subject + "/" + session)
        metadata.write_text(json.dumps(sorted(groups), indent=2), encoding="utf-8")
    return {"saved": True, "message": "Đã lưu ảnh có nhãn", "sample": str(path.relative_to(root))}


def dataset_status(root=IMAGE_ROOT):
    counts = {label: 0 for label in LABELS}
    groups = set()
    subjects = set()
    for file in root.glob("*/*/*/*.png"):
        subject, session, label, _ = file.relative_to(root).parts
        if label in counts:
            counts[label] += 1
            groups.add((subject, session))
            if subject != "public_digits":
                subjects.add(subject)
    proxies = sum(subject == "public_digits" for subject, _ in groups)
    return {"counts": counts, "total": sum(counts.values()), "sessions": len(groups)-proxies,
            "proxy_groups": proxies, "groups": len(groups), "subjects": len(subjects)}


def train_image_model(root=IMAGE_ROOT, output=IMAGE_MODEL, reports=IMAGE_REPORT, split_by="session"):
    """Group-disjoint split; choose candidates using validation, test once."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    from splitting import assign_groups_to_splits
    if split_by not in ("session", "subject"):
        raise ValueError("split_by phải là session hoặc subject")
    protocol_file = root / "dataset_protocol.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8")) if protocol_file.exists() else None
    if split_by == "subject" and any((root / "public_digits").glob("*/*/*.png")):
        raise ValueError("Ảnh công khai chưa có mã người; không thể công bố chia theo người. Chọn theo nhóm phiên/file hoặc bổ sung metadata người thật.")
    rows, seen = [], set()
    cropper = None
    cropped, fallback = 0, 0
    from image_preprocessing import CROP_VERSION, HandImageCropper
    cache = ROOT_DIR / "data" / "processed" / "image_crops" / CROP_VERSION
    for file in sorted(root.glob("*/*/*/*.png")):
        subject, session, label, _ = file.relative_to(root).parts
        if label not in LABELS:
            continue
        image = cv2.imread(str(file))
        if image is None:
            raise ValueError(f"Ảnh lỗi: {file.name}")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        group = subject if split_by == "subject" else subject + "/" + session
        if subject == "public_digits":
            cache.mkdir(parents=True, exist_ok=True)
            cached = cache / (digest + ".png")
            if cached.exists():
                image = cv2.imread(str(cached))
                if image is None:
                    raise ValueError("Ảnh cache bị lỗi: " + str(cached))
                # Store detection result with each cache item for reproducible counts.
                found = cached.with_suffix(".json").exists() and json.loads(cached.with_suffix(".json").read_text())["found"]
            else:
                if cropper is None:
                    cropper = HandImageCropper()
                image, found = cropper.crop(image)
                if not cv2.imwrite(str(cached), image):
                    raise ValueError("Không ghi được cache ảnh")
                cached.with_suffix(".json").write_text(json.dumps({"found": found}))
            cropped += int(found)
            fallback += int(not found)
        rows.append((file, image_features(image), label, group, image))
    if cropper:
        cropper.close()
    counts = {label: sum(row[2] == label for row in rows) for label in LABELS}
    if any(n < 30 for n in counts.values()):
        raise ValueError(f"Cần ít nhất 30 ảnh thật khác nhau cho mỗi tư thế; hiện có {counts}")
    metadata = root / "calibration_groups.json"
    calibration = set(json.loads(metadata.read_text(encoding="utf-8"))) if metadata.exists() else set()
    if split_by == "subject":
        calibration = {g.split("/")[0] for g in calibration}
    sizes = {g: sum(row[3] == g for row in rows) for g in {r[3] for r in rows} if g not in calibration}
    assignment = assign_groups_to_splits(sizes, seed=42)
    assignment.update({g: "train" for g in calibration if any(r[3] == g for r in rows)})
    parts = {split: [r for r in rows if assignment[r[3]] == split] for split in ("train", "val", "test")}
    for split, part in parts.items():
        missing = set(LABELS) - {r[2] for r in part}
        if missing:
            raise ValueError(f"Tập {split} thiếu {sorted(missing)}. Thu đủ 4 tư thế trong mỗi phiên/người.")
    X = {s: np.stack([r[1] for r in part]) for s, part in parts.items()}
    y = {s: np.array([r[2] for r in part]) for s, part in parts.items()}
    rng = np.random.default_rng(42)
    train_images, train_labels = [], []
    for row in parts["train"]:
        for variant in training_variants(row[4], rng):
            train_images.append(image_features(variant))
            train_labels.append(row[2])
    X["train"], y["train"] = np.stack(train_images), np.array(train_labels)
    # Validate both image orientations; these remain variants of the same val
    # samples, not additional independent examples and never enter model.fit.
    val_original = X["val"]
    val_mirror = np.stack([image_features(cv2.flip(r[4], 1)) for r in parts["val"]])
    val_labels = y["val"]
    X["val"] = np.concatenate([val_original, val_mirror])
    y["val"] = np.concatenate([val_labels, val_labels])
    candidates = [(f"SVM C={c}", make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", probability=True, random_state=42))) for c in (1., 10.)]
    candidates += [("KNN k=5", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(5, len(y['train'])), weights="distance"))),
                   ("Random Forest", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1))]
    validation, best, score = [], None, -1.
    for name, model in candidates:
        start = time.perf_counter()
        model.fit(X["train"], y["train"])
        pred = model.predict(X["val"])
        f1 = float(f1_score(y["val"], pred, average="macro"))
        validation.append({"model": name, "macro_f1": f1, "accuracy": float(accuracy_score(y["val"], pred)),
                           "original_accuracy": float(accuracy_score(val_labels, pred[:len(val_labels)])),
                           "mirrored_accuracy": float(accuracy_score(val_labels, pred[len(val_labels):])),
                           "fit_seconds": time.perf_counter()-start})
        if f1 > score:
            best, score = (name, model), f1
    name, model = best
    # Threshold selection uses validation only; display coverage explicitly.
    probabilities = model.predict_proba(X["val"])
    confidence = probabilities.max(axis=1)
    predicted = model.classes_[probabilities.argmax(axis=1)]
    threshold = .65
    for value in (.55, .60, .65, .70, .75, .80, .85):
        accepted = confidence >= value
        if accepted.mean() >= .5 and (predicted[accepted] == y["val"][accepted]).mean() >= .95:
            threshold = value
            break
    pred = model.predict(X["test"])
    test_prob = model.predict_proba(X["test"]).max(axis=1)
    accepted = test_prob >= threshold
    mirror_test = np.stack([image_features(cv2.flip(r[4], 1)) for r in parts["test"]])
    mirror_pred = model.predict(mirror_test)
    report = {"schema_version": 1, "feature_version": FEATURE_VERSION, "input": "RGB/BGR image pixels -> grayscale HOG (1764 features), no landmarks",
              "model": name, "samples": counts, "split_by": split_by, "dataset_protocol": protocol,
              "calibration_train_only_groups": sorted(g for g in assignment if g in calibration),
              "calibration_note": "Webcam calibration groups are training-only. Validation/test scores do not measure performance on these users/conditions.",
              "training_augmentation": {"scope": "train only, after group split", "original_train_images": len(parts["train"]),
                                        "effective_train_images": len(y["train"]), "variants": ["original", "horizontal_flip", "small_rotation_low_light_noise", "flipped_small_rotation_low_light_noise"]},
              "validation_protocol": "original + mirrored variants of the same validation images; select by combined macro F1, not test",
              "test_mirrored_accuracy": float(accuracy_score(y["test"], mirror_pred)),
              "test_mirrored_macro_f1": float(f1_score(y["test"], mirror_pred, average="macro")),
              "image_framing": {"version": CROP_VERSION, "source_hand_crops": cropped, "source_full_frame_fallbacks": fallback,
                                "collected_images": "already cropped by the same web bounding-box formula"},
              "split_counts": {s: len(p) for s, p in parts.items()}, "group_assignment": assignment,
              "validation": validation, "confidence_threshold": threshold,
              "test_accuracy": float(accuracy_score(y["test"], pred)),
              "test_macro_f1": float(f1_score(y["test"], pred, average="macro")),
              "test_coverage": float(accepted.mean()),
              "test_accepted_accuracy": float((pred[accepted] == y["test"][accepted]).mean()) if accepted.any() else None,
              "classification_report": classification_report(y["test"], pred, output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(y["test"], pred, labels=LABELS).tolist(), "labels": list(LABELS),
              "limitations": "Session-disjoint does not imply subject-disjoint. Similar frames within a session are not independent samples. Collect multiple people and sessions."}
    reports.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"feature_version": FEATURE_VERSION, "model": model, "threshold": threshold, "report": report}
    temporary = output.with_suffix(".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(output)
    (reports / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    import csv
    with (reports / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "group", "split"])
        writer.writerows((str(r[0].relative_to(root)), r[2], r[3], assignment[r[3]]) for r in rows)
    return report


class ImagePoseClassifier:
    def __init__(self, path=IMAGE_MODEL):
        self.path = path
        self.artifact = None
        self.error = None
        self.reload()

    def reload(self):
        self.artifact = None
        self.error = None
        if self.path.exists():
            try:
                artifact = joblib.load(self.path)
                if artifact.get("feature_version") != FEATURE_VERSION or set(artifact["model"].classes_) != set(LABELS):
                    raise ValueError("Mô hình ảnh không khớp phiên bản đặc trưng/nhãn")
                self.artifact = artifact
            except Exception as exc:
                self.error = str(exc)

    def status(self):
        return {"trained": self.artifact is not None, "error": self.error,
                "report": self.artifact["report"] if self.artifact else None}

    def predict(self, image):
        artifact = self.artifact
        if artifact is None:
            raise ValueError("Chưa có mô hình ảnh cho cử chỉ Morse. Thu ảnh rồi huấn luyện trước.")
        start = time.perf_counter()
        model = artifact["model"]
        probabilities = model.predict_proba([image_features(image)])[0]
        index = int(np.argmax(probabilities))
        label, confidence = str(model.classes_[index]), float(probabilities[index])
        accepted = confidence >= artifact["threshold"]
        # A confidently learned rest class unlocks the FSM; uncertain frames cancel it.
        pose = ("IDLE" if label == "UNKNOWN" else label) if accepted else "UNKNOWN"
        return {"pose": pose, "predicted_class": label, "confidence": confidence,
                "accepted": bool(accepted), "confidence_threshold": artifact["threshold"],
                "latency_ms": (time.perf_counter()-start)*1000,
                "probabilities": {str(c): float(p) for c, p in zip(model.classes_, probabilities)}}
