from scipy.ndimage import binary_dilation, gaussian_filter1d
from matplotlib import pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)


def get_hand_points(image):
    detector = vision.HandLandmarker.create_from_options(options)
    # image = cv2.imread(image_path)
    image_height, image_width, num_channels = image.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    # mp_image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(mp_image)
    if len(detection_result.hand_landmarks) == 0:
        return []
    hand_norm_points = np.array([(l.x, l.y) for h in detection_result.hand_landmarks for l in h if 0<l.x<1 and 0<l.y<1])
    image_size = np.array([image_width, image_height])
    return np.round(hand_norm_points * image_size.T).astype(int)

