import cv2 as cv
import numpy as np

from ..config import DetectionConfig
from .detectors import detect_eye, detect_face


def calculate_drowsiness_score(eye_results, yawn_results):
    eye_mean = np.mean(eye_results) if eye_results else 0
    yawn_mean = np.mean(yawn_results) if yawn_results else 0

    alertness = (10 * eye_mean + yawn_mean) / 6
    drowsiness = 100 - (alertness * 100)

    return max(0, min(100, drowsiness))


def predict_drowsiness(model_eye, model_yawn, video_source=0, frames=30):
    cap = cv.VideoCapture(video_source)
    eye_results = []
    yawn_results = []

    try:
        for _ in range(frames):
            ret, frame = cap.read()
            if not ret:
                break

            anchor = detect_eye(frame)
            if anchor is None:
                continue

            for _ in range(5):
                eye_input = detect_eye(frame)
                if eye_input is not None:
                    result = model_eye.predict([anchor, eye_input],
                                              verbose=0)[0][0]
                    eye_results.append(result)

                face_input = detect_face(frame)
                if face_input is not None:
                    region = frame[
                        DetectionConfig.YAWN_REGION_Y:
                        DetectionConfig.YAWN_REGION_Y + DetectionConfig.YAWN_REGION_H,
                        DetectionConfig.YAWN_REGION_X:
                        DetectionConfig.YAWN_REGION_X + DetectionConfig.YAWN_REGION_W
                    ]
                    yawn_input = detect_face(region)
                    if yawn_input is not None:
                        result = model_yawn.predict([face_input, yawn_input],
                                                    verbose=0)[0][0]
                        yawn_results.append(result)

    finally:
        cap.release()

    return calculate_drowsiness_score(eye_results, yawn_results)
