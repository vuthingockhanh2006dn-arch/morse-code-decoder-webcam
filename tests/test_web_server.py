"""Integration checks for the single Python pipeline used by the web UI."""
import json
import sys
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import web_server
from config import DEFAULT_CONFIG
from fsm_controller import MorseFSM


class EngineTest(unittest.TestCase):
    def setUp(self):
        with patch.object(web_server.MorseEngine, "_load_available_models"):
            self.engine = web_server.MorseEngine()

    def test_auto_selects_configured_best_model_and_heuristic_ignores_models(self):
        self.engine.cfg["best_model"] = "svm"
        for name in ("svm", "rf"):
            model = Mock()
            model.predict.return_value = ["PINCH_THUMB"]
            model.predict_proba.return_value = [[.98, .02]]
            self.engine.models[name] = model
        points = [[.5, .5, 0.] for _ in range(21)]
        points[0], points[9] = [.5, .8, 0.], [.5, .5, 0.]
        points[5], points[17] = [.35, .5, 0.], [.65, .5, 0.]
        points[4], points[8] = [.4, .3, 0.], [.42, .3, 0.]
        points[16], points[20] = [.7, .3, 0.], [.9, .3, 0.]
        self.engine.detector.stable_time = 0
        self.assertEqual(self.engine.predict_pose(points, algo="auto")[0], "PINCH_THUMB")
        self.engine.models["svm"].predict.assert_called_once()
        self.engine.models["rf"].predict.assert_not_called()
        self.engine.models["svm"].reset_mock()
        self.assertEqual(self.engine.predict_pose(points, algo="heuristic")[0], "PINCH_THUMB")
        self.engine.models["svm"].predict.assert_not_called()

    def test_camera_loss_cancels_gesture_preserving_text(self):
        self.engine.buffer.text = "A"
        self.engine.step_fsm("IDLE", 1., 0, camera=True)
        for i in range(1, 7):
            self.engine.step_fsm("PINCH_THUMB", 1., i/30, camera=True)
        self.engine.step_fsm(None, 1., .23, camera=True)
        self.assertEqual(self.engine.buffer.buffer, "")
        self.assertEqual(self.engine.buffer.text, "A")
        self.assertEqual(self.engine.fsm.state, "IDLE")

    def test_release_near_dash_boundary_stays_dot(self):
        fsm = MorseFSM(dict(DEFAULT_CONFIG, smoothing_window=1, release_frames=3))
        events = []
        fsm.update("PINCH_THUMB", 1., 0.)
        fsm.update("PINCH_THUMB", 1., .3)
        for t in (.38, .41, .44):
            events += fsm.update("IDLE", 1., t)
        self.assertEqual(events, ["DOT"])

    def test_unknown_and_missing_model_fail_cleanly(self):
        with self.assertRaises(ValueError):
            self.engine.predict_pose([[0, 0, 0]] * 21, algo="svm")
        with self.assertRaises(ValueError):
            self.engine.predict_pose([[float("nan"), 0, 0]] * 21)
        with self.assertRaises(ValueError):
            self.engine.step_fsm("BAD", 1.)


class HTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(web_server.MorseEngine, "_load_available_models"):
            cls.engine = web_server.MorseEngine()
        cls.patch = patch.object(web_server, "ENGINE", cls.engine)
        cls.patch.start()
        cls.server = HTTPServer(("127.0.0.1", 0), web_server.MorseRequestHandler)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.patch.stop()

    def post(self, path, data):
        request = Request(self.url + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=3) as response:
            return json.load(response)

    def test_assets_and_single_morse_dictionary(self):
        for path in ("/", "/app.js"):
            with urlopen(self.url + path, timeout=3) as response:
                self.assertEqual(response.status, 200)
        with urlopen(self.url + "/api/status", timeout=3) as response:
            self.assertEqual(len(json.load(response)["morse_table"]), 36)

    def test_malformed_request_and_directory_escape_are_rejected(self):
        for body in ([1, 2], {"landmarks": [["bad", 0, 0]] * 21}):
            with self.assertRaises(HTTPError) as error:
                self.post("/api/predict", body)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url + "/%2e%2e/requirements.txt", timeout=3)
        self.assertEqual(error.exception.code, 404)
        error.exception.close()

    def test_virtual_hold_release_and_double_tap_keep_previous_letters(self):
        self.post("/api/reset", {})
        clock = [0.]
        def feed(pose, frames):
            state = None
            for _ in range(frames):
                clock[0] += 1/30
                with patch.object(web_server.time, "perf_counter", return_value=clock[0]):
                    state = self.post("/api/fsm_step", {"pose": pose, "confidence": 1.})
            return state
        for _ in range(2):
            feed("PINCH_THUMB", 6)
            feed("IDLE", 8)
            feed("PINCH_RING", 5)
            feed("IDLE", 5)
            feed("PINCH_RING", 5)
            state = feed("IDLE", 8)
        self.assertEqual(state["decoded_text"], "EE")
        self.assertEqual(state["morse_buffer"], "")
        self.assertEqual(self.post("/api/cancel", {})["decoded_text"], "EE")
        self.assertEqual(self.post("/api/reset", {})["decoded_text"], "")


if __name__ == "__main__":
    unittest.main()
