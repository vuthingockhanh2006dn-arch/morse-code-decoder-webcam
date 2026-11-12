"""Local web UI: browser landmarks -> detector/models -> Python FSM -> MorseBuffer.

Run: python -X utf8 src/web_server.py [--port 8000]. No frontend decoder.
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

import config
from feature_engineering import extract_full_feature_vector
from fsm_controller import MorseFSM
from gesture_detector import GestureDetector
from morse_decoder import MORSE_TO_CHAR, MorseBuffer, decode_symbol

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class MorseEngine:
    def __init__(self):
        self.cfg = config.load_config()
        self.fsm = MorseFSM(self.cfg)
        self.buffer = MorseBuffer()
        self.detector = GestureDetector(
            touch=self.cfg["touch_threshold"], release=self.cfg["release_threshold"],
            ambiguity=self.cfg["ambiguity_margin"], stable_time=self.cfg["contact_stable_time"], anchor=4)
        self.models: Dict[str, Any] = {}
        self.ready = False
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
                "morse_buffer": self.buffer.buffer, "decoded_text": self.buffer.text,
                "preview": decode_symbol(self.buffer.buffer)}

    def cancel(self):
        """Cancel a pending gesture without deleting already entered symbols/text."""
        self.fsm.reset()
        self.detector.reset()
        self.ready = False
        self.last_hand = None
        self.last_algo = None
        return self.snapshot()

    def reset_fsm(self):
        self.cancel()
        self.buffer.clear()
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

    def step_fsm(self, pose: Optional[str], confidence: float, timestamp=None, camera=False):
        if pose not in (None, "IDLE", "UNKNOWN", "PINCH_THUMB", "PINCH_RING", "PINCH_PINKY"):
            raise ValueError("Tư thế không hợp lệ")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Độ tin cậy phải nằm trong [0, 1]")
        now = time.perf_counter() if timestamp is None else timestamp
        if not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("Thời gian không hợp lệ")
        if camera:
            if pose in (None, "UNKNOWN") or confidence < self.cfg["confidence_threshold"]:
                return self.cancel()
            if pose == "IDLE":
                self.ready = True
            if not self.ready:
                return self.snapshot()
        events = self.fsm.update(pose, confidence, now)
        for event in events:
            self.buffer.apply(event)
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
        if path == "/api/status":
            self._send_json(200, {"status": "online", "loaded_models": list(ENGINE.models),
                                  "config": ENGINE.cfg, "morse_table": MORSE_TO_CHAR, **ENGINE.snapshot()})
            return
        file = (WEB_DIR / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
        if not file.is_relative_to(WEB_DIR.resolve()) or not file.is_file():
            self.send_error(404)
            return
        content = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(str(file))[0] or "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self._send_json(403, {"error": "Chỉ chấp nhận lệnh từ giao diện cùng máy chủ"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768:
                raise ValueError("Kích thước yêu cầu không hợp lệ")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Yêu cầu phải là một JSON object")
            if self.path == "/api/predict":
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
                result = ENGINE.step_fsm(data.get("pose", "IDLE"), data.get("confidence", 1.))
            elif self.path == "/api/reset":
                result = ENGINE.reset_fsm()
            elif self.path == "/api/cancel":
                result = ENGINE.cancel()
            else:
                self._send_json(404, {"error": "Không có endpoint này"})
                return
            self._send_json(200, result)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self._send_json(400, {"error": str(exc)})


def run_server(port=8000):
    httpd = HTTPServer(("127.0.0.1", port), MorseRequestHandler)
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
