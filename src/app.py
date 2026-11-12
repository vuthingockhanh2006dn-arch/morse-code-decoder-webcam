"""Vietnamese desktop interface. Run: python src/app.py."""
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

from config import load_config
from fsm_controller import MorseFSM
from gesture_detector import GestureDetector
from morse_decoder import MorseBuffer, CHAR_TO_MORSE


class MorseApp:
    def __init__(self, root):
        self.root = root
        root.title("Morse Studio · Điều khiển bằng cử chỉ tay")
        root.geometry("1180x780")
        root.minsize(960, 700)
        root.configure(bg="#101827")
        self.cfg = load_config()
        self.buffer = MorseBuffer()
        self.messages = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.worker = None
        self.session = 0
        self.status = tk.StringVar(value="Sẵn sàng · Bật camera hoặc thử bằng các nút bên dưới")
        self.pose = tk.StringVar(value="Chưa có bàn tay")
        self.code = tk.StringVar(value="—")
        self.preview = tk.StringVar(value="Ký tự dự kiến: —")
        self.output = tk.StringVar(value="Văn bản sẽ xuất hiện ở đây")
        self.camera_index = tk.StringVar(value="0")
        self.touch = tk.DoubleVar(value=self.cfg["touch_threshold"])
        self.dash = tk.DoubleVar(value=self.cfg["dash_time"])
        self.progress = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#101827")
        style.configure("TLabel", background="#101827", foreground="#e5edf8", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11), padding=9)
        style.configure("Title.TLabel", font=("Segoe UI", 25, "bold"))
        style.configure("Code.TLabel", font=("Consolas", 36, "bold"), foreground="#55dfbd")
        main = ttk.Frame(root, padding=24)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        ttk.Label(main, text="MORSE STUDIO", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(main, text="Chạm ngón tay. Ghép tín hiệu. Viết thông điệp.").grid(row=1, column=0, sticky="w", pady=(2, 18))
        body = ttk.Frame(main)
        body.grid(row=2, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 22))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body)
        sidebar.grid(row=0, column=1, sticky="ns")
        canvas = tk.Canvas(sidebar, width=325, bg="#101827", highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        right = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=right, anchor="nw", width=325)
        right.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        def scroll_sidebar(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")
        canvas.bind("<MouseWheel>", scroll_sidebar)
        self.sidebar_canvas = canvas
        self.video = tk.Label(left, bg="#192638", fg="#91a7c2", text="WEBCAM\n\nĐưa một bàn tay vào khung hình",
                              font=("Segoe UI", 15), width=1, height=1)
        self.video.grid(row=0, column=0, sticky="nsew")
        ttk.Label(left, textvariable=self.pose, wraplength=550).grid(row=2, column=0, sticky="w", pady=(4, 4))
        ttk.Progressbar(left, variable=self.progress, maximum=1).grid(row=3, column=0, sticky="ew")
        controls = ttk.Frame(left)
        controls.grid(row=1, column=0, sticky="ew", pady=8)
        ttk.Label(controls, text="Camera:").pack(side="left")
        ttk.Entry(controls, textvariable=self.camera_index, width=3).pack(side="left", padx=6)
        self.start_button = ttk.Button(controls, text="Bật camera", command=self.start)
        self.start_button.pack(side="left", padx=4)
        ttk.Button(controls, text="Dừng", command=self.stop).pack(side="left", padx=4)
        ttk.Label(right, text="TÍN HIỆU ĐANG NHẬP").pack(anchor="w")
        ttk.Label(right, textvariable=self.code, style="Code.TLabel").pack(anchor="w", pady=8)
        ttk.Label(right, textvariable=self.preview).pack(anchor="w")
        ttk.Label(right, text="THÔNG ĐIỆP").pack(anchor="w", pady=(22, 10))
        ttk.Label(right, textvariable=self.output, wraplength=310,
                  font=("Segoe UI", 19, "bold")).pack(anchor="w")
        ttk.Label(right, text="CỬ CHỈ ĐIỀU KHIỂN").pack(anchor="w", pady=(24, 10))
        for text in ["• Dot: trỏ chạm cái nhanh rồi thả", "• Dash: trỏ chạm cái và giữ",
                     "• Space: trỏ chạm áp út 1 lần", "• Chốt chữ: trỏ chạm áp út 2 lần",
                     "• Delete: trỏ chạm út", "Thả tay về tư thế mở giữa các lệnh."]:
            ttk.Label(right, text=text, wraplength=315).pack(anchor="w", pady=3)
        ttk.Label(right, text="ĐIỀU CHỈNH · áp dụng khi bật lại camera").pack(anchor="w", pady=(20, 6))
        for label, variable, low, high in [("Ngưỡng chạm (0.12–0.40)", self.touch, .12, .4),
                                            ("Thời gian dash (0.25–1.00 giây)", self.dash, .25, 1.)]:
            ttk.Label(right, text=label).pack(anchor="w")
            tk.Scale(right, variable=variable, from_=low, to=high, resolution=.01, orient="horizontal",
                     bg="#101827", fg="#e5edf8", highlightthickness=0, length=300).pack(fill="x")
        buttons = ttk.Frame(main)
        buttons.grid(row=3, column=0, sticky="ew", pady=(12, 8))
        for label, event in [("Dot  ·", "DOT"), ("Dash  −", "DASH"), ("Chốt chữ", "CONFIRM"),
                              ("Space", "SPACE"), ("Delete", "DELETE")]:
            ttk.Button(buttons, text=label, command=lambda e=event: self.manual(e)).pack(side="left", padx=3)
        ttk.Button(buttons, text="Xóa tất cả", command=self.clear).pack(side="left", padx=3)
        ttk.Button(buttons, text="Sao chép", command=self.copy).pack(side="left", padx=3)
        ttk.Label(main, text="TRA MÃ   " + "   ".join(f"{c} {CHAR_TO_MORSE[c]}" for c in "AEHIMNOST"),
                  foreground="#91a7c2").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Label(main, textvariable=self.status, wraplength=850).grid(row=5, column=0, sticky="w")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(25, self.poll)

    def apply(self, event):
        self.buffer.apply(event)
        self.refresh()
        self.status.set(f"Đã nhận: {event}")

    def manual(self, event):
        if self.worker and self.worker.is_alive():
            self.stop()
        self.apply(event)

    def refresh(self):
        self.code.set(self.buffer.buffer or "—")
        self.preview.set("Ký tự dự kiến: " + (self.buffer.preview or "—"))
        self.output.set(self.buffer.text or "Văn bản sẽ xuất hiện ở đây")

    def clear(self):
        self.stop()
        self.buffer.clear()
        self.refresh()

    def copy(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.buffer.text)
        self.status.set("Đã sao chép văn bản đã chốt.")

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            index = int(self.camera_index.get())
            if index < 0:
                raise ValueError
        except ValueError:
            self.status.set("Chỉ số camera phải là số nguyên không âm, ví dụ 0 hoặc 1.")
            return
        cfg = dict(self.cfg, touch_threshold=self.touch.get(), dash_time=self.dash.get())
        self.stop_event = threading.Event()
        self.session += 1
        self.worker = threading.Thread(target=self.capture, args=(index, cfg, self.stop_event, self.session), daemon=True)
        self.start_button.configure(state="disabled")
        self.status.set("Đang mở camera…")
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.session += 1
        self.progress.set(0)
        self.pose.set("Camera đã dừng")
        self.status.set("Đã dừng camera. Có thể thử nhập bằng các nút.")

    def publish(self, message, stop):
        # Backpressure preserves commands; dropping frames could drop DELETE/DOT.
        while not stop.is_set():
            try:
                self.messages.put(message, timeout=.1)
                return
            except queue.Full:
                pass

    def capture(self, index, cfg, stop, session):
        cap = tracker = None
        try:
            import cv2
            from hand_tracker import HandTracker
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                raise RuntimeError("Không mở được webcam. Kiểm tra quyền camera và chỉ số thiết bị.")
            tracker = HandTracker()
            detector = GestureDetector(cfg["touch_threshold"], cfg["touch_threshold"] + .10)
            fsm = MorseFSM(cfg)
            last_hand = None
            ready = False
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Mất kết nối webcam. Dừng và bật lại camera.")
                frame = cv2.flip(frame, 1)
                hand = tracker.process(frame)
                now = time.monotonic()
                if hand is None or (last_hand is not None and hand.handedness != last_hand):
                    # Losing tracking must cancel a pending touch, never synthesize DOT.
                    fsm.reset()
                    detector.reset()
                    ready = False
                last_hand = hand.handedness if hand else None
                pose = detector.detect(hand.landmarks if hand else None, frame.shape[1], frame.shape[0])
                if pose == "UNKNOWN":
                    fsm.reset()
                    ready = False
                if pose == "IDLE":
                    ready = True
                events = fsm.update(pose, 1., now) if hand and ready else []
                if hand:
                    tracker.draw(frame, hand)
                frame = cv2.resize(frame, (640, 480))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ppm = b"P6\n640 480\n255\n" + rgb.tobytes()
                distances = "  ".join(f"{k.removeprefix('PINCH_')}: {v:.2f}" for k, v in detector.distances.items())
                self.publish((session, "frame", (ppm, pose, events, fsm.hold_progress, distances)), stop)
        except Exception as exc:
            self.publish((session, "error", str(exc)), stop)
        finally:
            if tracker:
                tracker.close()
            if cap is not None:
                cap.release()

    def poll(self):
        try:
            self.drain_messages()
        except Exception as exc:
            self.stop()
            self.status.set(f"Lỗi hiển thị: {exc}. Có thể tiếp tục dùng các nút nhập.")
        finally:
            if not self.worker or not self.worker.is_alive():
                self.start_button.configure(state="normal")
            self.root.after(33, self.poll)

    def drain_messages(self):
        latest = None
        for _ in range(4):
            try:
                session, kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if session != self.session:
                continue
            if kind == "error":
                self.status.set("Lỗi: " + payload)
            else:
                ppm, pose, events, progress, distances = payload
                latest = ppm, pose, progress, distances
                for event in events:
                    self.apply(event)
        if latest:
            ppm, pose, progress, distances = latest
            self.photo = tk.PhotoImage(data=ppm, format="PPM")
            self.video.configure(image=self.photo, text="", width=1, height=1)
            self.pose.set(f"{pose or 'Không thấy bàn tay'}  ·  {distances}")
            self.progress.set(progress)

    def close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    MorseApp(tk.Tk()).root.mainloop()
