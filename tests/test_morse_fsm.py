"""Kiểm thử máy trạng thái và bộ giải mã Morse (không cần webcam hay thư viện ngoài).

Chạy: python3 -m unittest discover tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DEFAULT_CONFIG  # noqa: E402
from fsm_controller import MorseFSM  # noqa: E402
from morse_decoder import MorseBuffer, decode_message, encode_message  # noqa: E402

FPS = 30
DT = 1.0 / FPS


class Clock:
    t = 0.0


def feed(fsm, clock, pose, seconds):
    """Đưa `pose` vào FSM trong `seconds` giây ở 30 FPS, trả về các sự kiện."""
    events = []
    for _ in range(max(1, round(seconds * FPS))):
        clock.t += DT
        events += fsm.update(pose, 1.0, clock.t)
    return events


class MorseDecoderTest(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(decode_message(encode_message("SOS HELLO 2026")), "SOS HELLO 2026")

    def test_buffer(self):
        b = MorseBuffer()
        for ev in ["DOT", "DOT", "DOT", "CONFIRM", "DASH", "DASH", "DASH", "CONFIRM",
                   "DOT", "DOT", "DOT", "CONFIRM", "SPACE", "DOT", "DELETE", "DELETE"]:
            b.apply(ev)
        self.assertEqual(b.text, "SOS")
        self.assertEqual(b.buffer, "")

    def test_unknown_code_becomes_question_mark(self):
        b = MorseBuffer()
        for ev in ["DOT", "DOT", "DASH", "DASH", "CONFIRM"]:
            b.apply(ev)
        self.assertEqual(b.text, "?")

    def test_space_confirms_pending_symbols(self):
        b = MorseBuffer()
        for ev in ["DOT", "SPACE"]:
            b.apply(ev)
        self.assertEqual(b.text, "E ")

    def test_buffer_is_capped(self):
        b = MorseBuffer()
        for _ in range(8):
            b.apply("DOT")
        self.assertEqual(b.buffer, "." * 5)


class MorseFSMTest(unittest.TestCase):
    def setUp(self):
        self.fsm = MorseFSM(DEFAULT_CONFIG)
        self.clock = Clock()
        feed(self.fsm, self.clock, "UNKNOWN", 0.3)

    def gesture(self, pose, hold, rest=0.3):
        return feed(self.fsm, self.clock, pose, hold) + feed(self.fsm, self.clock, "UNKNOWN", rest)

    def test_dot(self):
        self.assertEqual(self.gesture("PINCH_THUMB", 0.2), ["DOT"])

    def test_dash_fires_once_while_holding(self):
        self.assertEqual(self.gesture("PINCH_THUMB", 3.0), ["DASH"])

    def test_space_single_ring_tap(self):
        self.assertEqual(self.gesture("PINCH_RING", 0.2, rest=0.8), ["SPACE"])

    def test_confirm_double_ring_tap(self):
        events = feed(self.fsm, self.clock, "PINCH_RING", 0.2)
        events += feed(self.fsm, self.clock, "UNKNOWN", 0.2)
        events += feed(self.fsm, self.clock, "PINCH_RING", 0.2)
        events += feed(self.fsm, self.clock, "UNKNOWN", 0.8)
        self.assertEqual(events, ["CONFIRM"])

    def test_slow_second_tap_is_a_new_space(self):
        events = self.gesture("PINCH_RING", 0.2, rest=0.8) + self.gesture("PINCH_RING", 0.2, rest=0.8)
        self.assertEqual(events, ["SPACE", "SPACE"])

    def test_delete_once_while_holding(self):
        self.assertEqual(self.gesture("PINCH_PINKY", 2.0), ["DELETE"])

    def test_locked_state_needs_rest_pose_to_unlock(self):
        feed(self.fsm, self.clock, "PINCH_THUMB", 1.0)           # DASH, rồi khóa
        self.assertEqual(feed(self.fsm, self.clock, "PINCH_THUMB", 1.0), [])
        feed(self.fsm, self.clock, "UNKNOWN", 0.3)               # nghỉ, mở khóa
        self.assertEqual(self.gesture("PINCH_THUMB", 0.2), ["DOT"])

    def test_no_hand_produces_nothing(self):
        self.assertEqual(feed(self.fsm, self.clock, None, 2.0), [])

    def test_single_frame_glitch_is_ignored(self):
        events = feed(self.fsm, self.clock, "PINCH_THUMB", DT) + feed(self.fsm, self.clock, "UNKNOWN", 0.5)
        self.assertEqual(events, [])

    def test_low_confidence_is_ignored(self):
        events = []
        for _ in range(60):
            self.clock.t += DT
            events += self.fsm.update("PINCH_THUMB", 0.3, self.clock.t)
        self.assertEqual(events, [])

    def test_type_sos(self):
        b = MorseBuffer()

        def confirm():
            return (feed(self.fsm, self.clock, "PINCH_RING", 0.15) + feed(self.fsm, self.clock, "UNKNOWN", 0.15)
                    + feed(self.fsm, self.clock, "PINCH_RING", 0.15) + feed(self.fsm, self.clock, "UNKNOWN", 0.6))

        events = []
        for hold in [0.2] * 3:
            events += self.gesture("PINCH_THUMB", hold)
        events += confirm()
        for hold in [0.8] * 3:
            events += self.gesture("PINCH_THUMB", hold)
        events += confirm()
        for hold in [0.2] * 3:
            events += self.gesture("PINCH_THUMB", hold)
        events += confirm()
        for ev in events:
            b.apply(ev)
        self.assertEqual(b.text, "SOS")


if __name__ == "__main__":
    unittest.main()
