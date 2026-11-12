"""UI regressions: keep controls reachable and render errors recoverable."""
import queue
import sys
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from app import MorseApp


class AppUITest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.app = MorseApp(self.root)
        self.root.geometry("960x700")
        self.root.update()

    def tearDown(self):
        if hasattr(self, "root"):
            self.root.destroy()

    def widgets(self, widget):
        for child in widget.winfo_children():
            yield child
            yield from self.widgets(child)

    def test_buttons_visible_at_minimum_window_size(self):
        buttons = [w for w in self.widgets(self.root) if w.winfo_class() == "TButton"]
        self.assertEqual(len(buttons), 9)
        for button in buttons:
            with self.subTest(button=button.cget("text")):
                self.assertTrue(button.winfo_ismapped())
                self.assertLessEqual(button.winfo_rooty() + button.winfo_height(),
                                     self.root.winfo_rooty() + self.root.winfo_height())
                self.assertLessEqual(button.winfo_rootx() + button.winfo_width(),
                                     self.root.winfo_rootx() + self.root.winfo_width())

    def test_manual_button_stops_capture_and_accepts_command(self):
        self.app.worker = Mock()
        self.app.worker.is_alive.return_value = True
        old_session = self.app.session
        button = next(w for w in self.widgets(self.root)
                      if w.winfo_class() == "TButton" and w.cget("text") == "Dot  ·")
        button.invoke()
        self.assertEqual(self.app.buffer.buffer, ".")
        self.assertTrue(self.app.stop_event.is_set())
        self.assertGreater(self.app.session, old_session)

    def test_frame_batch_preserves_events_and_draws_only_latest_frame(self):
        self.app.messages = queue.Queue()
        for event in ["DOT", "DASH", "CONFIRM"]:
            self.app.messages.put((self.app.session, "frame", (b"frame", "IDLE", [event], 0, "")))
        with patch("app.tk.PhotoImage", return_value="" ) as photo:
            self.app.drain_messages()
        self.assertEqual(self.app.buffer.text, "A")
        self.assertEqual(photo.call_count, 1)

    def test_render_failure_keeps_polling_and_manual_input_available(self):
        with patch.object(self.app, "drain_messages", side_effect=RuntimeError("bad image")), \
                patch.object(self.root, "after") as after:
            self.app.poll()
        after.assert_called_once_with(33, self.app.poll)
        self.app.manual("DOT")
        self.assertEqual(self.app.buffer.buffer, ".")


if __name__ == "__main__":
    unittest.main()
