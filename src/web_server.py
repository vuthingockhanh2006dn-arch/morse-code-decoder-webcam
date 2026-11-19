"""Local web UI: browser landmarks -> detector/models -> Python FSM -> MorseBuffer.

Run: python -X utf8 src/web_server.py [--port 8000]. No frontend decoder.
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import socket
import time
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

import config
from feature_engineering import extract_full_feature_vector
from fsm_controller import MorseFSM
from gesture_detector import GestureDetector
from morse_decoder import MORSE_TO_CHAR, MorseBuffer, decode_symbol
from image_learning import ImagePoseClassifier, decode_image, save_sample, dataset_status, train_image_model
from image_shape_learning import ImageShapeClassifier, train_shape_model

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class MorseEngine:
    def __init__(self):
        self.cfg = config.load_config()
        # Image inference is slower than landmarks: avoid accumulating five
        # frames of lag for a short tap. Keep confidence rejection unchanged.
        self.cfg.update(smoothing_window=3, release_frames=2, double_tap_window=.65, dash_time=.65)
        self.fsm = MorseFSM(self.cfg)
        self.buffer = MorseBuffer()
        self.detector = GestureDetector(
            touch=self.cfg["touch_threshold"], release=self.cfg["release_threshold"],
            ambiguity=self.cfg["ambiguity_margin"], stable_time=self.cfg["contact_stable_time"], anchor=4)
        self.models: Dict[str, Any] = {}
        self.image_classifier = ImagePoseClassifier()
        self.shape_classifier = ImageShapeClassifier()
        self.training = {"state": "idle", "message": "Chưa huấn luyện trong phiên này"}
        self.training_thread = None
        self.input_source = None
        self.image_uncertain_since = None
        self.image_missing_since = None
        self.upload_cropper = None
        self.ready = False
        self.last_event = None
        self.event_count = 0
        self.image_frames = 0
        self.arm_requests = 0
        self.last_cancel_reason = None
        self.last_api_error = None
        self.image_trace = deque(maxlen=18000)
        self.last_hand = None
        self.last_algo = None
        self._load_available_models()

    def _load_available_models(self):
        import joblib
        for algo in config.ALGORITHMS:
            path = config.model_path(algo)
            if path.exists():
                try:
                    self.models[algo] = joblib.load(path)
                except Exception as exc:
                    print(f"Không nạp được {path.name}: {exc}")

    def snapshot(self, events=None):
        return {"state": self.fsm.state, "events": events or [],
                "ready": self.ready,
                "last_event": self.last_event, "event_count": self.event_count,
                "morse_buffer": self.buffer.buffer, "decoded_text": self.buffer.text,
                "preview": decode_symbol(self.buffer.buffer)}

    def cancel(self, reason="cancel_requested"):
        """Cancel a pending gesture without deleting already entered symbols/text."""
        self.fsm.reset()
        self.last_cancel_reason = reason
        self.detector.reset()
        self.ready = False
        self.image_uncertain_since = None
        self.image_missing_since = None
        self.last_hand = None
        self.last_algo = None
        return self.snapshot()

    def interrupt_image_gesture(self, reason):
        """Discard an uncertain gesture without disabling the camera session.

        A command already emitted stays locked until a confirmed rest, so an
        uncertain frame cannot turn a held DELETE into repeated deletions.
        """
        ready, locked = self.ready, self.fsm.state == "LOCKED"
        self.cancel(reason)
        self.ready = ready
        if locked:
            self.fsm._go("LOCKED")
        return self.snapshot()

    def missing_image_hand(self, timestamp=None):
        """Tolerate short tracking dropouts without treating them as release."""
        now = time.perf_counter() if timestamp is None else timestamp
        if self.image_missing_since is None:
            self.image_missing_since = now
        if now - self.image_missing_since > .35:
            since = self.image_missing_since
            state = self.interrupt_image_gesture("hand_not_detected")
            self.image_missing_since = since
            return {**state, "recognition_hint": "Mất dấu tay: bỏ cử chỉ dở, giữ trạng thái mở khóa. Đưa tay trở lại camera; nếu vừa nhập lệnh, mở tay trước lệnh tiếp theo."}
        return {**self.snapshot(), "recognition_hint": "Tạm mất dấu tay; đang chờ camera bắt lại, chưa nhập lệnh."}

    def step_image(self, prediction, timestamp=None):
        """Brief uncertain transition frames freeze input; sustained loss cancels.

        Missing hand frames use a short tracking grace period in the API.
        No command is inferred from a rejected image.
        """
        now = time.perf_counter() if timestamp is None else timestamp
        if self.image_missing_since is not None:
            gap = now - self.image_missing_since
            self.image_missing_since = None
            if gap > .35:
                self.interrupt_image_gesture("hand_tracking_gap")
            else:
                # A dropout is neither a release nor time spent holding a pose.
                self.fsm.touch_start += gap
                self.fsm.first_tap_release += gap
                self.fsm.smoother.reset()
        if not prediction["accepted"]:
            if self.image_uncertain_since is None:
                self.image_uncertain_since = now
            if now-self.image_uncertain_since > .20:
                state = self.interrupt_image_gesture("image_confidence_timeout")
                return {**state, "recognition_hint": "Ảnh chuyển tiếp chưa rõ: đã bỏ cử chỉ dở, vẫn giữ mở khóa nhập. Mở tay rồi thử lại."}
            return {**self.snapshot(), "recognition_hint": "Đang chuyển tư thế, tạm giữ trạng thái; chưa nhập lệnh."}
        if self.image_uncertain_since is not None:
            gap = now-self.image_uncertain_since
            self.image_uncertain_since = None
            if gap > .20:
                self.interrupt_image_gesture("image_confidence_gap")
            else:
                # Do not count uncertain image intervals as a held DASH.
                self.fsm.touch_start += gap
                self.fsm.first_tap_release += gap
                self.fsm.smoother.reset()
        state = self.step_fsm(prediction["pose"], prediction["confidence"], now, camera=True,
                              confidence_floor=prediction["confidence_threshold"])
        hints = {"IDLE": "Sẵn sàng. Chạm cái–trỏ rồi thả để nhập dot; giữ để nhập dash.",
                 "THUMB_DOWN": "Đã nhận cái–trỏ. Thả để tạo dot hoặc tiếp tục giữ để tạo dash.",
                 "RING_DOWN": "Đã nhận cái–áp út. Thả ngón để hoàn tất lần chạm.",
                 "WAIT_SECOND": "Đang chờ chạm áp út lần hai để chốt chữ; chạm đơn sẽ thêm dấu cách.",
                 "LOCKED": "Đã nhập một lần. Mở tay trước lệnh tiếp theo."}
        hint = "Chưa mở khóa nhập. Mở bàn tay hoặc bấm Mở khóa nhập khi camera đang chạy." if not self.ready else hints[self.fsm.state]
        return {**state, "recognition_hint": hint}

    def image_status(self):
        return {**self.image_classifier.status(), "shape_report": self.shape_classifier.artifact['report'] if self.shape_classifier.artifact else None,
                "shape_error": self.shape_classifier.error, "dataset": dataset_status(), "training": dict(self.training)}

    def arm_input(self, source="image"):
        if source not in ("image", "landmark"):
            raise ValueError("Nguồn nhập không hợp lệ")
        self.cancel()
        self.input_source = source
        self.ready = True
        self.arm_requests += 1
        return {**self.snapshot(), "recognition_hint": "Đã mở khóa nhập. Chạm cái–trỏ rồi thả để nhập dot, hoặc giữ để nhập dash."}

    def train_images(self, split_by="session"):
        if split_by not in ("session", "subject"):
            raise ValueError("Cách chia tập không hợp lệ")
        if self.training_thread and self.training_thread.is_alive():
            raise ValueError("Đang huấn luyện, vui lòng chờ")
        self.training = {"state": "running", "message": "Đang trích HOG và huấn luyện SVM/KNN/Random Forest…"}
        def work():
            try:
                report = train_image_model(split_by=split_by)
                shape_report = train_shape_model(report)
                self.shape_classifier.reload()
                self.image_classifier.reload()
                if not self.image_classifier.status()["trained"]:
                    raise ValueError("Không nạp được mô hình vừa ghi: " + str(self.image_classifier.error))
                if self.shape_classifier.artifact is None:
                    raise ValueError("Không nạp được mô hình hình dạng: " + str(self.shape_classifier.error))
                self.training = {"state": "done", "message": f"Đã huấn luyện {shape_report['model']} từ ảnh · test macro F1 {shape_report['test_macro_f1']:.3f}"}
            except Exception as exc:
                self.training = {"state": "error", "message": str(exc)}
        self.training_thread = threading.Thread(target=work, daemon=True)
        self.training_thread.start()
        return self.image_status()

    def reset_fsm(self):
        self.cancel()
        self.buffer.clear()
        self.last_event = None
        self.event_count = 0
        return self.snapshot()

    def predict_pose(self, landmarks: List[List[float]], handedness="Right", algo="auto",
                     width=640, height=480, now=None) -> Tuple[str, float, float]:
        if algo not in ("auto", "heuristic", *config.ALGORITHMS):
            raise ValueError("Thuật toán không hợp lệ")
        if handedness not in ("Left", "Right"):
            raise ValueError("Nhãn tay không hợp lệ")
        if (not isinstance(landmarks, list) or len(landmarks) != 21 or
                any(not isinstance(p, list) or len(p) != 3 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p)
                    for p in landmarks)):
            raise ValueError("Cần 21 điểm [x, y, z] hữu hạn")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or
               not 1 <= v <= 8192 for v in (width, height)):
            raise ValueError("Kích thước khung hình không hợp lệ")
        if self.last_hand != handedness or self.last_algo != algo:
            self.cancel()
        self.last_hand, self.last_algo = handedness, algo
        start = time.perf_counter()
        pose = self.detector.detect(landmarks, width, height, now=now)
        target = None
        if algo == "auto":
            best = self.cfg.get("best_model")
            target = best if best in self.models else next((a for a in config.ALGORITHMS if a in self.models), None)
        elif algo != "heuristic":
            if algo not in self.models:
                raise ValueError(f"Chưa có mô hình {algo.upper()}; chọn Tự động hoặc Hình học")
            target = algo
        confidence = 1.0
        if target and pose not in (None, "IDLE", "UNKNOWN"):
            model = self.models[target]
            features = extract_full_feature_vector(landmarks, handedness)
            predicted = str(model.predict([features])[0])
            if hasattr(model, "predict_proba"):
                confidence = float(max(model.predict_proba([features])[0]))
            # Reject conflicts instead of changing fingers while in contact.
            if predicted != pose:
                pose, confidence = "UNKNOWN", 0.
        return pose, confidence, (time.perf_counter() - start) * 1000

    def step_fsm(self, pose: Optional[str], confidence: float, timestamp=None, camera=False, confidence_floor=None):
        if pose not in (None, "IDLE", "UNKNOWN", "PINCH_THUMB", "PINCH_RING", "PINCH_PINKY"):
            raise ValueError("Tư thế không hợp lệ")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Độ tin cậy phải nằm trong [0, 1]")
        now = time.perf_counter() if timestamp is None else timestamp
        if not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("Thời gian không hợp lệ")
        if camera:
            floor = self.cfg["confidence_threshold"] if confidence_floor is None else confidence_floor
            if pose in (None, "UNKNOWN") or confidence < floor:
                return self.cancel()
            if pose == "IDLE":
                self.ready = True
            if not self.ready:
                return self.snapshot()
        # Image confidence was already filtered at its validation-selected floor.
        events = self.fsm.update(pose, 1.0 if confidence_floor is not None else confidence, now)
        for event in events:
            self.buffer.apply(event)
            self.last_event = event
            self.event_count += 1
        return self.snapshot(events)


ENGINE = MorseEngine()


class MorseRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if args and str(args[0]).startswith("POST"):
            return
        super().log_message(format, *args)

    def _send_json(self, status, data):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/diagnostics/history":
            self._send_json(200, {"frames": list(ENGINE.image_trace),
                                  "capacity": ENGINE.image_trace.maxlen})
            return
        if path == "/api/status":
            self._send_json(200, {"status": "online", "loaded_models": list(ENGINE.models),
                                  "config": ENGINE.cfg, "morse_table": MORSE_TO_CHAR,
                                  "image_learning": ENGINE.image_status(),
                                  "diagnostics": {"image_frames": ENGINE.image_frames, "arm_requests": ENGINE.arm_requests,
                                    "last_cancel_reason": ENGINE.last_cancel_reason, "last_api_error": ENGINE.last_api_error,
                                    "recent_frames": list(ENGINE.image_trace)[-12:]}, **ENGINE.snapshot()})
            return
        file = (WEB_DIR / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
        if not file.is_relative_to(WEB_DIR.resolve()) or not file.is_file():
            self.send_error(404)
            return
        content = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(str(file))[0] or "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self._send_json(403, {"error": "Chỉ chấp nhận lệnh từ giao diện cùng máy chủ"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 650000:
                raise ValueError("Kích thước yêu cầu không hợp lệ")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Yêu cầu phải là một JSON object")
            if self.path == "/api/predict_image":
                ENGINE.image_frames += 1
                if ENGINE.input_source != "image":
                    ENGINE.cancel()
                    ENGINE.input_source = "image"
                if not data.get("hand_present", True):
                    result = {"pose": None, "confidence": 0., "latency_ms": 0., **ENGINE.missing_image_hand()}
                else:
                    image = decode_image(data.get("image"))
                    if data.get("crop_input", False):
                        from image_preprocessing import HandImageCropper
                        if ENGINE.upload_cropper is None:
                            ENGINE.upload_cropper = HandImageCropper()
                        image, _ = ENGINE.upload_cropper.crop(image)
                    if data.get("landmarks") is not None and ENGINE.shape_classifier.artifact is not None:
                        prediction = ENGINE.shape_classifier.predict(data['landmarks'], data.get('width',128), data.get('height',128))
                    else:
                        prediction = ENGINE.image_classifier.predict(image)
                        prediction['recognition_model'] = 'HOG image (fallback)'
                    apply = data.get("apply", True)
                    if not isinstance(apply, bool):
                        raise ValueError("apply phải là true/false")
                    result = {**prediction, **(ENGINE.step_image(prediction) if apply else ENGINE.snapshot())}
                ENGINE.image_trace.append({"time": round(time.time(), 3), "pose": result.get("pose"),
                    "predicted_class": result.get("predicted_class"), "confidence": round(result.get("confidence", 0), 3),
                    "probabilities": result.get("probabilities", {}),
                    "last_cancel_reason": ENGINE.last_cancel_reason,
                    "accepted": result.get("accepted", False), "ready": result["ready"], "state": result["state"],
                    "events": result["events"], "morse_buffer": result["morse_buffer"], "reason": result.get("recognition_hint")})
            elif self.path == "/api/images/collect":
                if ENGINE.training["state"] == "running":
                    raise ValueError("Dừng thu ảnh trong lúc huấn luyện để cố định dữ liệu")
                result = {**save_sample(data.get("image"), data.get("label"), data.get("subject"), data.get("session"),
                                       train_only=data.get("train_only", False)),
                          "dataset": dataset_status()}
            elif self.path == "/api/images/train":
                result = ENGINE.train_images(data.get("split_by", "session"))
            elif self.path == "/api/predict":
                if ENGINE.input_source != "landmark":
                    ENGINE.cancel()
                    ENGINE.input_source = "landmark"
                landmarks = data.get("landmarks")
                if landmarks is None:
                    result = {"pose": None, "confidence": 0., "latency_ms": 0., **ENGINE.cancel()}
                else:
                    pose, conf, latency = ENGINE.predict_pose(landmarks, data.get("handedness", "Right"),
                        data.get("algo", "auto"), data.get("width", 640), data.get("height", 480))
                    distances = dict(ENGINE.detector.distances)
                    result = {"pose": pose, "confidence": conf, "latency_ms": latency,
                              "distances": distances,
                              **ENGINE.step_fsm(pose, conf, camera=True)}
            elif self.path == "/api/fsm_step":
                if ENGINE.input_source != "simulator":
                    ENGINE.cancel()
                    ENGINE.input_source = "simulator"
                result = ENGINE.step_fsm(data.get("pose", "IDLE"), data.get("confidence", 1.))
            elif self.path == "/api/reset":
                result = ENGINE.reset_fsm()
            elif self.path == "/api/arm":
                result = ENGINE.arm_input(data.get("source", "image"))
            elif self.path == "/api/cancel":
                result = ENGINE.cancel()
            else:
                self._send_json(404, {"error": "Không có endpoint này"})
                return
            self._send_json(200, result)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            ENGINE.last_api_error = str(exc)
            self._send_json(400, {"error": str(exc)})


class LocalHTTPServer(HTTPServer):
    # Windows SO_REUSEADDR can let a second server bind the same address and
    # serve a stale version unpredictably. Own the listener exclusively.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def run_server(port=8000):
    httpd = LocalHTTPServer(("127.0.0.1", port), MorseRequestHandler)
    print(f"Morse Studio: http://localhost:{port} · Ctrl+C để dừng")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Giao diện web Morse")
    parser.add_argument("--port", type=int, default=8000)
    run_server(parser.parse_args().port)
