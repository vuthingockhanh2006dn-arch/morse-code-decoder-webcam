"""Image pipeline tests. Synthetic fixtures are temporary, not research data."""
import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from image_learning import (LABELS, FEATURE_VERSION, ImagePoseClassifier, dataset_status,
                            decode_image, image_features, save_sample, train_image_model, training_variants)
from import_pose_images import import_images, POSE_MAPPING
from image_preprocessing import crop_box


def encoded(image):
    return base64.b64encode(cv2.imencode(".png", image)[1]).decode()


class ImageLearningTest(unittest.TestCase):
    def test_augmentation_includes_mirror_without_mutating_source(self):
        image = np.zeros((64,64,3), np.uint8)
        image[:, :15] = 200
        original = image.copy()
        variants = training_variants(image, np.random.default_rng(42))
        self.assertEqual(len(variants), 4)
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(variants[1], cv2.flip(image, 1))
        self.assertTrue(all(v.shape == image.shape and v.dtype == np.uint8 for v in variants))
    def test_training_crop_matches_browser_formula_and_stays_in_frame(self):
        points = [(0.1, 0.2), (0.9, 0.8)]
        x, y, side = crop_box(points, 640, 480)
        expected_side = min(640, 480, max(.8*640, .6*480)*1.45)
        self.assertEqual(side, expected_side)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x+side, 640)
        self.assertLessEqual(y+side, 480)

    def test_existing_pose_images_import_with_provenance_and_proxy_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/"source", Path(tmp)/"target"
            for digit in POSE_MAPPING:
                folder = source / digit
                folder.mkdir(parents=True)
                image = np.full((100, 100, 3), int(digit)*20, np.uint8)
                cv2.imwrite(str(folder/f"IMG_{1118+int(digit)}.JPG"), image)
            report = import_images(source, target)
            self.assertEqual(sum(report["counts"].values()), 4)
            self.assertEqual(len(list(target.glob("*/*/*/*.png"))), 4)
            import_images(source, target)
            self.assertEqual(len(list(target.glob("*/*/*/*.png"))), 4)
            status = dataset_status(target)
            self.assertEqual(status["subjects"], 0)
            self.assertEqual(status["sessions"], 0)
            self.assertEqual(status["proxy_groups"], 1)
            self.assertTrue((target/"source_manifest.csv").exists())
            with self.assertRaises(ValueError):
                train_image_model(target, Path(tmp)/"model.joblib", Path(tmp)/"report", split_by="subject")

    def test_image_decode_hog_and_invalid_inputs(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.rectangle(image, (30, 20), (90, 110), (200, 200, 200), -1)
        restored = decode_image(encoded(image))
        self.assertEqual(restored.shape, image.shape)
        self.assertEqual(image_features(restored).shape, (1764,))
        self.assertTrue(np.isfinite(image_features(restored)).all())
        for bad in (None, "not base64", base64.b64encode(b"not an image").decode()):
            with self.assertRaises(ValueError):
                decode_image(bad)

    def test_collection_paths_labels_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = encoded(np.full((128, 128, 3), 100, dtype=np.uint8))
            self.assertTrue(save_sample(image, "UNKNOWN", "P01", "S01", root)["saved"])
            self.assertFalse(save_sample(image, "PINCH_RING", "P02", "S02", root)["saved"])
            self.assertEqual(dataset_status(root)["total"], 1)
            with self.assertRaises(ValueError):
                save_sample(image, "DOT", "P01", "S01", root)
            with self.assertRaises(ValueError):
                save_sample(image, "UNKNOWN", "../escape", "S01", root)

    def test_calibration_collection_marks_only_explicit_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = encoded(np.full((128,128,3), 80, np.uint8))
            save_sample(image, "UNKNOWN", "webcam01", "calib01", root, train_only=True)
            groups = json.loads((root/"calibration_groups.json").read_text())
            self.assertEqual(groups, ["webcam01/calib01"])

    def test_missing_real_data_cannot_publish_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "model.joblib"
            with self.assertRaises(ValueError):
                train_image_model(Path(tmp)/"empty", output, Path(tmp)/"report")
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                ImagePoseClassifier(output).predict(np.zeros((64,64,3), np.uint8))

    def test_train_save_reload_and_disjoint_groups_end_to_end(self):
        # Isolated synthetic patterns verify mechanics only. Nothing is saved to models/.
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"images"
            for subject in range(4):
                for index, label in enumerate(LABELS):
                    folder = root / f"P{subject}" / "S01" / label
                    folder.mkdir(parents=True)
                    for i in range(10):
                        image = rng.integers(0, 12, (64,64,3), dtype=np.uint8)
                        if index == 0:
                            cv2.line(image, (8,15), (55,15), (240,240,240), 9)
                        elif index == 1:
                            cv2.line(image, (15,8), (15,55), (240,240,240), 9)
                        elif index == 2:
                            cv2.circle(image, (32,32), 15, (240,240,240), 5)
                        else:
                            cv2.line(image, (8,8), (55,55), (240,240,240), 9)
                        cv2.imwrite(str(folder/f"{i}.png"), image)
            (root/"calibration_groups.json").write_text(json.dumps(["P3/S01"]))
            output, reports = Path(tmp)/"model.joblib", Path(tmp)/"reports"
            report = train_image_model(root, output, reports, split_by="subject")
            self.assertEqual(sum(report["split_counts"].values()), 160)
            self.assertEqual(report["group_assignment"]["P3"], "train")
            self.assertEqual(report["calibration_train_only_groups"], ["P3"])
            augmentation = report["training_augmentation"]
            self.assertEqual(augmentation["effective_train_images"], augmentation["original_train_images"]*4)
            self.assertIn("after group split", augmentation["scope"])
            self.assertEqual(set(report["group_assignment"].values()), {"train", "val", "test"})
            self.assertEqual(len(report["validation"]), 4)
            self.assertEqual(report["feature_version"], FEATURE_VERSION)
            self.assertTrue((reports/"split_manifest.csv").exists())
            self.assertEqual(json.loads((reports/"metrics.json").read_text(encoding="utf-8"))["labels"], list(LABELS))
            classifier = ImagePoseClassifier(output)
            self.assertTrue(classifier.status()["trained"])
            prediction = classifier.predict(image)
            self.assertIn(prediction["predicted_class"], LABELS)
            self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.)


if __name__ == "__main__":
    unittest.main()
