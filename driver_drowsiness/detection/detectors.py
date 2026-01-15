import cv2 as cv
import numpy as np
import tensorflow as tf

from ..config import HaarCascadeConfig, ModelArchitecture

_face_cascade = None
_eye_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv.CascadeClassifier(
            str(HaarCascadeConfig.FACE_CASCADE_PATH)
        )
    return _face_cascade


def _get_eye_cascade():
    global _eye_cascade
    if _eye_cascade is None:
        _eye_cascade = cv.CascadeClassifier(
            str(HaarCascadeConfig.EYE_CASCADE_PATH)
        )
    return _eye_cascade


def detect_region(image, cascade, scale_factor, min_neighbors, size):
    regions = cascade.detectMultiScale(
        image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
    )
    if len(regions) == 0:
        return None

    (x, y, w, h) = regions[0]
    region = image[y:y+h, x:x+w]
    region = cv.cvtColor(region, cv.COLOR_BGR2RGB)
    region = cv.resize(region, (size, size))
    region = region / 255.0
    region = tf.expand_dims(region, axis=0)
    return region


def detect_eye(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return detect_region(
        gray,
        _get_eye_cascade(),
        HaarCascadeConfig.SCALE_FACTOR,
        HaarCascadeConfig.MIN_NEIGHBORS,
        ModelArchitecture.IMAGE_SIZE,
    )


def detect_face(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return detect_region(
        gray,
        _get_face_cascade(),
        HaarCascadeConfig.SCALE_FACTOR,
        HaarCascadeConfig.MIN_NEIGHBORS,
        ModelArchitecture.IMAGE_SIZE,
    )
