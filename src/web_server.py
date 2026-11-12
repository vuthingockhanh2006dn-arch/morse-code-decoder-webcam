"""Máy chủ web cục bộ phục vụ giao diện tương tác và API dự đoán Morse.

Tính năng:
  * Phục vụ static web từ thư mục `web/` (HTML, CSS, JS).
  * API `GET /api/status`: Kiểm tra trạng thái và danh sách mô hình đã nạp.
  * API `POST /api/predict`: Nhận 21 landmark từ webcam trình duyệt, tính 66 đặc trưng
    và trả về nhãn tư thế (PINCH_THUMB, PINCH_RING, PINCH_PINKY, UNKNOWN), độ tin cậy và độ trễ (ms).
  * API `POST /api/fsm_step`: Đưa tư thế vào FSM và trả về sự kiện, chuỗi Morse, văn bản giải mã.

Chạy:
    python3 src/web_server.py [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
WEB_DIR = ROOT_DIR / "web"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from feature_engineering import (
    compute_additional_distances,
    extract_full_feature_vector,
)
from fsm_controller import MorseFSM
from morse_decoder import MorseBuffer


# --------------------------------------------------------------------------- #
# Bộ quản lý mô hình và máy trạng thái trong bộ nhớ
# --------------------------------------------------------------------------- #
class MorseEngine:
    def __init__(self) -> None:
        self.cfg = config.load_config()
        self.fsm = MorseFSM(self.cfg)
        self.buffer = MorseBuffer()
        self.models: Dict[str, Any] = {}
        self._load_available_models()

    def _load_available_models(self) -> None:
        for algo in config.ALGORITHMS:
            model_path = config.model_path(algo)
            if model_path.exists():
                try:
                    import joblib
                    self.models[algo] = joblib.load(model_path)
                    print(f"[*] Đã nạp mô hình: {algo} từ {model_path.name}")
                except Exception as e:
                    print(f"[!] Không thể nạp mô hình {algo}: {e}")

    def predict_pose(
        self, landmarks: List[List[float]], handedness: str = "Right", algo: str = "auto"
    ) -> Tuple[str, float, float]:
        """Trả về (pose, confidence, latency_ms)."""
        t0 = time.perf_counter()
        features = extract_full_feature_vector(landmarks, handedness=handedness)

        # 1. Nếu có mô hình scikit-learn tương ứng
        target_algo = algo if algo in self.models else ("rf" if "rf" in self.models else None)
        if target_algo and target_algo in self.models:
            model = self.models[target_algo]
            pose = str(model.predict([features])[0])
            conf = 1.0
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([features])[0]
                conf = float(max(probs))
            t1 = time.perf_counter()
            return pose, conf, (t1 - t0) * 1000.0

        # 2. Quy tắc heuristic theo khoảng cách Euclid ngón tay
        # Đặc trưng 63, 64, 65 là 3 khoảng cách Euclid
        d_thumb = features[-3]
        d_ring = features[-2]
        d_pinky = features[-1]
        threshold = 0.38

        pose = "UNKNOWN"
        conf = 0.95
        if d_thumb < threshold and d_thumb < min(d_ring, d_pinky):
            pose = "PINCH_THUMB"
        elif d_ring < threshold and d_ring < min(d_thumb, d_pinky):
            pose = "PINCH_RING"
        elif d_pinky < threshold and d_pinky < min(d_thumb, d_ring):
            pose = "PINCH_PINKY"
        else:
            conf = 0.70

        t1 = time.perf_counter()
        return pose, conf, (t1 - t0) * 1000.0

    def step_fsm(self, pose: str, confidence: float, timestamp: Optional[float] = None) -> Dict[str, Any]:
        now = timestamp if timestamp is not None else time.perf_counter()
        events = self.fsm.update(pose, confidence, now)
        for e in events:
            self.buffer.apply(e)

        return {
            "state": self.fsm.state,
            "events": events,
            "morse_buffer": self.buffer.buffer,
            "decoded_text": self.buffer.text,
        }

    def reset_fsm(self) -> None:
        self.fsm.reset()
        self.buffer = MorseBuffer()


ENGINE = MorseEngine()


# --------------------------------------------------------------------------- #
# Request Handler
# --------------------------------------------------------------------------- #
class MorseRequestHandler(BaseHTTPRequestHandler):
    server_version = "MorseServer/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Giảm ồn console, chỉ log lỗi hoặc api
        if "/api/" in self.path:
            super().log_message(format, *args)

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._send_json(200, {
                "status": "online",
                "loaded_models": list(ENGINE.models.keys()),
                "config": ENGINE.cfg,
            })
            return

        # Phục vụ file tĩnh từ web/
        file_path = WEB_DIR / ("index.html" if self.path in ("/", "") else self.path.lstrip("/"))
        if file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"
            try:
                content = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, f"Error reading file: {e}")
        else:
            self.send_error(404, "File not found")

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len)

        try:
            req_data = json.loads(post_body.decode("utf-8")) if post_body else {}
        except Exception:
            self._send_json(400, {"error": "Invalid JSON payload"})
            return

        if self.path == "/api/predict":
            landmarks = req_data.get("landmarks")
            if not landmarks or len(landmarks) != 21:
                self._send_json(400, {"error": "Cần mảng 21 điểm landmarks [x, y, z]"})
                return

            handedness = req_data.get("handedness", "Right")
            algo = req_data.get("algo", "auto")
            pose, conf, latency = ENGINE.predict_pose(landmarks, handedness=handedness, algo=algo)

            # Tự động đẩy qua FSM nếu client yêu cầu
            fsm_res = {}
            if req_data.get("step_fsm", True):
                fsm_res = ENGINE.step_fsm(pose, conf, req_data.get("timestamp"))

            res = {
                "pose": pose,
                "confidence": round(conf, 4),
                "latency_ms": round(latency, 2),
                **fsm_res,
            }
            self._send_json(200, res)

        elif self.path == "/api/fsm_step":
            pose = req_data.get("pose", "UNKNOWN")
            conf = float(req_data.get("confidence", 1.0))
            ts = req_data.get("timestamp")
            res = ENGINE.step_fsm(pose, conf, ts)
            self._send_json(200, res)

        elif self.path == "/api/reset":
            ENGINE.reset_fsm()
            self._send_json(200, {"message": "Đã đặt lại FSM và bộ đệm Morse"})

        else:
            self.send_error(404, "Endpoint not found")


def run_server(port: int = 8000) -> None:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    server_address = ("", port)
    httpd = HTTPServer(server_address, MorseRequestHandler)
    print(f"\n=======================================================")
    print(f" [OK] Máy chủ Web Morse đang hoạt động tại:")
    print(f"      http://localhost:{port}")
    print(f"      http://127.0.0.1:{port}")
    print(f" Nhấn Ctrl+C để dừng máy chủ.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Đang tắt máy chủ web...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy máy chủ web test Morse")
    parser.add_argument("--port", type=int, default=8000, help="Cổng web (mặc định: 8000)")
    args = parser.parse_args()
    run_server(args.port)
