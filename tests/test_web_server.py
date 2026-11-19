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

    def test_image_threshold_is_not_replaced_by_landmark_threshold(self):
        self.engine.step_fsm("IDLE", .60, 0., camera=True, confidence_floor=.55)
        for i in range(1, 8):
            self.engine.step_fsm("PINCH_THUMB", .60, i/30, camera=True, confidence_floor=.55)
        self.assertEqual(self.engine.fsm.state, "THUMB_DOWN")

    def test_arm_preserves_text_and_enables_image_events_without_rest_classifier(self):
        self.engine.buffer.text = "A"
        state = self.engine.arm_input("image")
        self.assertTrue(state["ready"])
        self.assertEqual(state["decoded_text"], "A")
        self.assertEqual(self.engine.input_source, "image")
        prediction = {"pose": "PINCH_THUMB", "accepted": True, "confidence": .9, "confidence_threshold": .55}
        for i in range(16):
            state = self.engine.step_image(prediction, i/20)
        self.assertEqual(state["morse_buffer"], "-")
        self.assertEqual(state["last_event"], "DASH")
        self.assertEqual(state["event_count"], 1)

    def test_unarmed_pose_changes_do_not_create_commands(self):
        prediction = {"pose": "PINCH_THUMB", "accepted": True, "confidence": .9, "confidence_threshold": .55}
        for i in range(20):
            state = self.engine.step_image(prediction, i/20)
        self.assertFalse(state["ready"])
        self.assertEqual(state["morse_buffer"], "")
        self.assertEqual(state["event_count"], 0)

    def test_brief_tracking_dropout_preserves_arm_and_does_not_create_event(self):
        self.engine.arm_input("image")
        self.engine.fsm = MorseFSM(dict(DEFAULT_CONFIG, smoothing_window=1, release_frames=1))
        good = {"pose": "PINCH_THUMB", "accepted": True, "confidence": .9, "confidence_threshold": .55}
        self.engine.step_image(good, 0.)
        self.assertTrue(self.engine.missing_image_hand(.1)["ready"])
        self.assertEqual(self.engine.missing_image_hand(.2)["events"], [])
        state = self.engine.step_image(dict(good, pose="IDLE"), .3)
        self.assertEqual(state["events"], ["DOT"])
        self.assertTrue(state["ready"])

    def test_sustained_tracking_loss_cancels_pending_gesture(self):
        self.engine.arm_input("image")
        self.engine.missing_image_hand(0.)
        state = self.engine.missing_image_hand(.4)
        self.assertTrue(state["ready"])
        self.assertEqual(state["events"], [])

    def test_tracking_loss_keeps_delete_locked_until_real_rest(self):
        self.engine.arm_input("image")
        self.engine.fsm = MorseFSM(dict(DEFAULT_CONFIG, smoothing_window=1, release_frames=1))
        self.engine.buffer.text = "ABC"
        pred = {"pose": "PINCH_PINKY", "accepted": True, "confidence": .9, "confidence_threshold": .55}
        self.engine.step_image(pred, 0.)
        self.engine.missing_image_hand(.1)
        self.engine.missing_image_hand(.6)
        state = self.engine.step_image(pred, .8)
        self.assertTrue(state["ready"])
        self.assertEqual(state["decoded_text"], "AB")
        self.assertEqual(state["events"], [])
        self.engine.step_image(dict(pred, pose="IDLE"), .9)
        state = self.engine.step_image(pred, 1.)
        self.assertEqual(state["decoded_text"], "A")

    def test_half_second_image_tap_is_dot_and_hold_emits_one_dash(self):
        self.engine.arm_input("image")
        events = []
        for i in range(11):
            events += self.engine.step_image({"pose": "PINCH_THUMB", "accepted": True,
                "confidence": .9, "confidence_threshold": .55}, i*.05)["events"]
        for i in range(4):
            events += self.engine.step_image({"pose": "IDLE", "accepted": True,
                "confidence": .9, "confidence_threshold": .55}, .55+i*.05)["events"]
        self.assertEqual(events, ["DOT"])
        for i in range(40):
            events += self.engine.step_image({"pose": "PINCH_THUMB", "accepted": True,
                "confidence": .9, "confidence_threshold": .55}, 1.+i*.05)["events"]
        self.assertEqual(events, ["DOT", "DASH"])

    def test_image_short_tap_and_double_tap_at_ten_fps(self):
        now = 0.
        events = []
        def feed(pose, frames):
            nonlocal now
            state = None
            for _ in range(frames):
                now += .1
                state = self.engine.step_image({"pose": pose, "accepted": True,
                    "confidence": .9, "confidence_threshold": .55}, now)
                events.extend(state["events"])
            return state
        feed("IDLE", 3)
        feed("PINCH_THUMB", 3)
        state = feed("IDLE", 3)
        self.assertEqual(state["morse_buffer"], ".")
        feed("PINCH_RING", 3)
        feed("IDLE", 3)
        feed("PINCH_RING", 3)
        state = feed("IDLE", 3)
        self.assertEqual(state["decoded_text"], "E")
        self.assertEqual(events, ["DOT", "CONFIRM"])

    def test_rest_release_does_not_inherit_smoothed_thumb_duration(self):
        fsm = MorseFSM(dict(DEFAULT_CONFIG, smoothing_window=5, release_frames=2))
        for t in (0., .04, .08):
            fsm.update("IDLE", 1., t)
        for t in (.12, .16, .20, .24, .28):
            fsm.update("PINCH_THUMB", 1., t)
        events = []
        for t in (.5, .6):
            events += fsm.update("IDLE", 1., t)
        self.assertEqual(events, ["DOT"])

    def test_ring_release_with_lagging_filter_does_not_restart_or_add_space(self):
        fsm = MorseFSM(dict(DEFAULT_CONFIG, smoothing_window=5, release_frames=2))
        fsm.update("PINCH_RING", 1., 0.)
        for t in (.04, .08, .12, .16):
            fsm.update("PINCH_RING", 1., t)
        self.assertEqual(fsm.update("IDLE", 1., .20), [])
        self.assertEqual(fsm.update("IDLE", 1., .24), [])
        self.assertEqual(fsm.state, "WAIT_SECOND")
        self.assertEqual(fsm.update("IDLE", 1., .28), [])
        self.assertEqual(fsm.state, "WAIT_SECOND")

    def test_confirmed_rest_unlocks_without_waiting_for_filter_to_flush(self):
        fsm = MorseFSM(dict(DEFAULT_CONFIG, smoothing_window=5, release_frames=2))
        for i in range(20):
            fsm.update("PINCH_THUMB", 1., i*.05)
        self.assertEqual(fsm.state, "LOCKED")
        self.assertEqual(fsm.update("IDLE", 1., 1.), [])
        self.assertEqual(fsm.state, "LOCKED")
        self.assertEqual(fsm.update("IDLE", 1., 1.05), [])
        self.assertEqual(fsm.state, "IDLE")

    def test_brief_uncertain_image_transition_does_not_erase_dot(self):
        self.engine.fsm = MorseFSM(dict(DEFAULT_CONFIG, smoothing_window=1, release_frames=1))
        def pred(pose, accepted=True):
            return {"pose": pose, "accepted": accepted, "confidence": .9 if accepted else .3,
                    "confidence_threshold": .55}
        self.engine.step_image(pred("IDLE"), 0.)
        self.engine.step_image(pred("PINCH_THUMB"), .1)
        self.assertEqual(self.engine.step_image(pred("UNKNOWN", False), .2)["events"], [])
        state = self.engine.step_image(pred("IDLE"), .25)
        self.assertEqual(state["events"], ["DOT"])
        self.assertEqual(state["morse_buffer"], ".")

    def test_sustained_uncertain_image_cancels_without_phantom_dot(self):
        self.engine.fsm = MorseFSM(dict(DEFAULT_CONFIG, smoothing_window=1, release_frames=1))
        good = {"pose": "IDLE", "accepted": True, "confidence": .9, "confidence_threshold": .55}
        self.engine.step_image(good, 0.)
        self.engine.step_image(dict(good, pose="PINCH_THUMB"), .1)
        bad = dict(good, pose="UNKNOWN", accepted=False, confidence=.3)
        self.engine.step_image(bad, .2)
        state = self.engine.step_image(bad, .5)
        self.assertTrue(state["ready"])
        self.assertEqual(state["morse_buffer"], "")

    def test_uncertain_transition_does_not_relock_session_or_repeat_delete(self):
        self.engine.fsm = MorseFSM(dict(DEFAULT_CONFIG, smoothing_window=1, release_frames=1))
        self.engine.arm_input("image")
        self.engine.buffer.text = "ABC"
        good = {"pose": "PINCH_PINKY", "accepted": True, "confidence": .9, "confidence_threshold": .55}
        self.engine.step_image(good, 0.)
        bad = dict(good, pose="UNKNOWN", accepted=False, confidence=.3)
        self.engine.step_image(bad, .1)
        state = self.engine.step_image(bad, .4)
        self.assertTrue(state["ready"])
        self.assertEqual(state["state"], "LOCKED")
        state = self.engine.step_image(good, .5)
        self.assertEqual(state["decoded_text"], "AB")
        self.assertEqual(state["events"], [])
        self.engine.step_image(dict(good, pose="IDLE"), .6)
        state = self.engine.step_image(good, .7)
        self.assertEqual(state["decoded_text"], "A")

    def test_rejected_frame_cannot_repeat_a_locked_delete(self):
        self.engine.fsm = MorseFSM(dict(DEFAULT_CONFIG, smoothing_window=1, release_frames=1))
        self.engine.buffer.text = "AN"
        good = {"pose": "IDLE", "accepted": True, "confidence": .9, "confidence_threshold": .55}
        self.engine.step_image(good, 0.)
        self.engine.step_image(dict(good, pose="PINCH_PINKY"), .1)
        self.engine.step_image(dict(good, pose="UNKNOWN", accepted=False, confidence=.3), .2)
        state = self.engine.step_image(dict(good, pose="PINCH_PINKY"), .25)
        self.assertEqual(state["decoded_text"], "A")
        self.assertEqual(state["events"], [])

    def test_failed_training_reports_error_without_creating_fake_model(self):
        with patch.object(web_server, "train_image_model", side_effect=ValueError("Need real images")):
            self.engine.train_images()
            self.engine.training_thread.join(timeout=3)
        self.assertEqual(self.engine.training["state"], "error")
        self.assertIn("Need real images", self.engine.training["message"])

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
            status = json.load(response)
            self.assertEqual(len(status["morse_table"]), 36)
            self.assertIn("image_learning", status)
            self.assertIn("trained", status["image_learning"])

    def test_image_api_arm_to_dash_returns_symbol_to_frontend(self):
        import base64
        import cv2
        import numpy as np
        self.post("/api/reset", {})
        state = self.post("/api/arm", {"source": "image"})
        self.assertTrue(state["ready"])
        start_count = state["event_count"]
        image = base64.b64encode(cv2.imencode(".png", np.full((128,128,3), 100, np.uint8))[1]).decode()
        pred = {"pose": "PINCH_THUMB", "predicted_class": "PINCH_THUMB", "accepted": True,
                "confidence": .9, "confidence_threshold": .55, "latency_ms": 1.}
        with patch.object(self.engine.image_classifier, "predict", return_value=pred):
            for i in range(16):
                with patch.object(web_server.time, "perf_counter", return_value=i/20):
                    state = self.post("/api/predict_image", {"image": image})
        self.assertEqual(state["morse_buffer"], "-")
        self.assertEqual(state["last_event"], "DASH")
        self.assertEqual(state["event_count"], start_count+1)

    def test_image_api_requires_real_model_and_preserves_text_on_hand_loss(self):
        self.engine.arm_input("image")
        self.engine.buffer.text = "AN"
        with self.assertRaises(HTTPError) as error:
            self.post("/api/predict_image", {"image": "not an image", "apply": False})
        self.assertEqual(error.exception.code, 400)
        error.exception.close()
        with patch.object(web_server.time, "perf_counter", return_value=10.):
            state = self.post("/api/predict_image", {"hand_present": False})
        self.assertEqual(state["events"], [])
        with patch.object(web_server.time, "perf_counter", return_value=10.4):
            state = self.post("/api/predict_image", {"hand_present": False})
        self.assertEqual(state["decoded_text"], "AN")
        self.assertEqual(state["state"], "IDLE")

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
