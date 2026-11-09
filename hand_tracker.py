"""MediaPipe Tasks adapter for the existing gesture/drawing loop."""
import time
from types import SimpleNamespace
from model_asset import verify_model


class HandTracker:
    def __init__(self, model_path):
        verify_model(model_path)
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.last_timestamp = -1
        self.landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            ))

    def process(self, rgb):
        self.last_timestamp = max(self.last_timestamp + 1, time.monotonic_ns() // 1000000)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, self.last_timestamp)
        return SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=points)
                                                    for points in result.hand_landmarks])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.landmarker.close()
