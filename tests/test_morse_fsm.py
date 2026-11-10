"""Unit tests for the parts that need no camera: Morse lookup and the FSM.

Run with:  python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DEFAULT_CONFIG  # noqa: E402
from fsm_controller import MorseFSM, SignCommitter  # noqa: E402
from morse_decoder import MorseBuffer, decode_message, encode_message  # noqa: E402

FPS = 30
DT = 1.0 / FPS


class Clock:
    def __init__(self):
        self.t = 0.0


def feed(fsm, clock, pose, seconds):
    """Feed `pose` for `seconds` at 30 FPS and collect emitted events."""
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

    def test_unknown_code(self):
        b = MorseBuffer()
        for ev in ["DOT", "DOT", "DASH", "DASH", "CONFIRM"]:
            b.apply(ev)
        self.assertEqual(b.text, "?")


class MorseFSMTest(unittest.TestCase):
    def setUp(self):
        self.fsm = MorseFSM(DEFAULT_CONFIG)
        self.clock = Clock()
        feed(self.fsm, self.clock, "IDLE", 0.3)

    def gesture(self, pose, hold, rest=0.3):
        return feed(self.fsm, self.clock, pose, hold) + feed(self.fsm, self.clock, "IDLE", rest)

    def test_dot(self):
        self.assertEqual(self.gesture("PINCH_THUMB", 0.2), ["DOT"])

    def test_dash_fires_once_while_holding(self):
        self.assertEqual(self.gesture("PINCH_THUMB", 3.0), ["DASH"])

    def test_space_single_ring_tap(self):
        self.assertEqual(self.gesture("PINCH_RING", 0.2, rest=0.8), ["SPACE"])

    def test_confirm_double_ring_tap(self):
        events = feed(self.fsm, self.clock, "PINCH_RING", 0.2)
        events += feed(self.fsm, self.clock, "IDLE", 0.2)
        events += feed(self.fsm, self.clock, "PINCH_RING", 0.2)
        events += feed(self.fsm, self.clock, "IDLE", 0.8)
        self.assertEqual(events, ["CONFIRM"])

    def test_delete_once_while_holding(self):
        self.assertEqual(self.gesture("PINCH_PINKY", 2.0), ["DELETE"])

    def test_no_hand_produces_nothing(self):
        self.assertEqual(feed(self.fsm, self.clock, None, 2.0), [])

    def test_single_frame_glitch_is_ignored(self):
        events = feed(self.fsm, self.clock, "PINCH_THUMB", DT) + feed(self.fsm, self.clock, "IDLE", 0.5)
        self.assertEqual(events, [])

    def test_type_sos(self):
        b = MorseBuffer()
        seq = [("PINCH_THUMB", 0.2)] * 3 + [("RING2", 0)] + [("PINCH_THUMB", 0.8)] * 3 + \
              [("RING2", 0)] + [("PINCH_THUMB", 0.2)] * 3 + [("RING2", 0)]
        for pose, hold in seq:
            if pose == "RING2":
                evs = feed(self.fsm, self.clock, "PINCH_RING", 0.15) + feed(self.fsm, self.clock, "IDLE", 0.15) \
                    + feed(self.fsm, self.clock, "PINCH_RING", 0.15) + feed(self.fsm, self.clock, "IDLE", 0.6)
            else:
                evs = self.gesture(pose, hold)
            for ev in evs:
                b.apply(ev)
        self.assertEqual(b.text, "SOS")


class SignCommitterTest(unittest.TestCase):
    def test_commit_once_until_rest(self):
        c = SignCommitter(DEFAULT_CONFIG)
        out = [c.update("A", 0.9) for _ in range(60)]
        self.assertEqual([x for x in out if x], ["A"])
        for _ in range(10):
            c.update("NOTHING", 0.9)
        out = [c.update("A", 0.9) for _ in range(60)]
        self.assertEqual([x for x in out if x], ["A"])

    def test_low_confidence_ignored(self):
        c = SignCommitter(DEFAULT_CONFIG)
        self.assertFalse(any(c.update("A", 0.3) for _ in range(60)))


if __name__ == "__main__":
    unittest.main()
