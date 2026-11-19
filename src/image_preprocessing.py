"""Same hand framing as web/app.js, used for source images and uploaded photos."""
import cv2

CROP_VERSION = "mediapipe-square-1.45-v1"


def crop_box(points, width, height):
    xs, ys = [p[0]*width for p in points], [p[1]*height for p in points]
    side = max(32., min(width, height, max(max(xs)-min(xs), max(ys)-min(ys))*1.45))
    x = max(0., min(width-side, (min(xs)+max(xs)-side)/2))
    y = max(0., min(height-side, (min(ys)+max(ys)-side)/2))
    return x, y, side


class HandImageCropper:
    def __init__(self):
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
            min_detection_confidence=.5, model_complexity=1)

    def crop(self, image):
        # Upscale small public 100x100 photographs for MediaPipe localization.
        h, w = image.shape[:2]
        large = cv2.resize(image, (round(w*400/max(w,h)), round(h*400/max(w,h))))
        result = self.hands.process(cv2.cvtColor(large, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return cv2.resize(image, (128,128)), False
        points = [(p.x, p.y) for p in result.multi_hand_landmarks[0].landmark]
        h, w = large.shape[:2]
        x, y, side = crop_box(points, w, h)
        roi = large[round(y):round(y+side), round(x):round(x+side)]
        return cv2.resize(roi, (128,128)), True

    def close(self):
        self.hands.close()
