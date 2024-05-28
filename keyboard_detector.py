from lib.chm_image.keyboard_store import KeyboardStore
from lib.sm_image.utils import imshow
import cv2
import numpy as np
import json
from lib.sm_image.key_points import harris_key_point_positions, cv2_key_points
from lib.sm_image.linear_model import LinearModel, ImageGradientCounter
from lib.sm_image.utils import draw_points
from lib.sm_image.color_model import ColorModel, GaussianModel, GaussianModelBinaryClassifier
from lib.chm_image.hand import get_hand_points
from matplotlib import pyplot as plt
from lib.sm_utils.lib.list_and_dict import dict_elem_with_default
from .keyboard_main_segment_candidate import KeyboardMainSegmentCandidate, DetectionParams, KeyboardMainSegmentParams



class KeyboardDetectionParams(DetectionParams):
    def __init__(self, params=None):
        super().__init__(params)
        self.hand_color_innovation_threshold = dict_elem_with_default(params, 'hand_color_innovation_threshold', 2.5)
        self.along_distance_threshold = dict_elem_with_default(params, 'along_distance_threshold', 100)
        self.along_distance_ratio_threshold = dict_elem_with_default(params, 'along_distance_ratio_threshold', .06)
        self.lm_theta_res = dict_elem_with_default(params, 'lm_theta_res', .1)
        main_segment_params = dict_elem_with_default(params, 'main_segment', {})
        self.main_segment = KeyboardMainSegmentParams(main_segment_params)
        
            

class KeyboardDetector:
    def __init__(self, detection_params=None):
        self.store = KeyboardStore(None)
        self.params = KeyboardDetectionParams(detection_params)
        self.main_segment_model = None

    def load_main_segment_model(self):
        with open('training_descriptors.json', 'r') as f:
            d = json.load(f)
        training_descriptors = np.array(d).T[:,:-1]
        # print(training_histograms.shape, training_histograms)
        # print(training_histograms.var(axis=1))
        self.main_segment_model = GaussianModelBinaryClassifier(training_descriptors)
    
    def set_image(self, image):
        self.image = image
        self.image_height, self.image_width, num_channels = image.shape
        self.grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def detect_keypoints(self):
        keypoint_locs = harris_key_point_positions(self.image)
        keypoints = cv2_key_points(keypoint_locs)
        self.keypoint_locs = np.array([(float(x[1]), float(x[0])) for x in keypoint_locs])

    def detect_hand(self):
        self.hand_points = get_hand_points(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.hand_cm = ColorModel(self.image)
        self.hand_cm.set_colors_at_points(self.hand_points)
        self.hand_cm.drop_outliers(innovation_threshold=2)

    def calculate_linear_model(self):
        lm = LinearModel(self.keypoint_locs, counter_class=ImageGradientCounter, counter_kwargs={'image': self.image}, 
                         theta_res=self.params.lm_theta_res)
        lm.set_hough_lines(threshold_ratio=.5, eps=10)
        self.segments = lm.line_segments(along_distance_threshold=self.params.along_distance_threshold,
                                         along_distance_ratio_threshold=self.params.along_distance_ratio_threshold)
        self.main_segment_candidates = []
        for segment in self.segments:
            segment.image_size = (self.image_height, self.image_width)
            candidate = KeyboardMainSegmentCandidate(self, segment)
            candidate.sort_endpoints()
            self.main_segment_candidates.append(candidate)
        self.lm = lm

    def plot_segments(self):
        draw_image = self.image.copy()
        for candidate in self.main_segment_candidates:
            candidate.plot(draw_image, plot_sides=True)
        imshow(draw_image)
            
    def calculate_side_models(self):
        for candidate in self.main_segment_candidates:
            candidate.calculate_side_models()
            
    def select_main_segment(self):
        scores = [candidate.score() for candidate in self.main_segment_candidates]
        self.lm.selected_line = np.argmax(scores)

    def to_dict(self):
        candidates = [c.to_dict() for c in self.main_segment_candidates]
        return {
            'main_segment': {
                'candidates': candidates,
                'selected_index': int(self.lm.selected_line)
                }
        }
    
    @staticmethod
    def from_dict(d, image):
        detector = KeyboardDetector()
        detector.load_main_segment_model()
        detector.set_image(image)
        detector.detect_keypoints()
        detector.detect_hand()
        detector.lm = LinearModel(detector.keypoint_locs, counter_class=ImageGradientCounter, 
                                  counter_kwargs={'image': image}, theta_res=detector.params.lm_theta_res, fit=False)
        detector.lm.selected_line = d['main_segment']['selected_index']
        detector.main_segment_candidates = []
        for candidate_dict in d['main_segment']['candidates']:
            detector.main_segment_candidates.append(KeyboardMainSegmentCandidate.from_dict(candidate_dict, detector))
        detector.calculate_side_models()
        return detector

    def detect(self, image, main_segment_model=None):
        if main_segment_model is None:
            self.load_main_segment_model()
        else:
            self.main_segment_model = main_segment_model
        self.set_image(image)
        self.detect_keypoints()
        self.detect_hand()
        self.calculate_linear_model()
        self.calculate_side_models()
        self.select_main_segment()
        
        # training_descriptors_list.append(sift_descriptors(grayscale_image, keypoints))
        # new_training_labels = np.zeros((len(keypoint_locs), 1))

        # draw_points(draw_image, hand_points, (128, 128, 255), radius=10, thickness=10)
        
        # plt.figure()
        # imshow(draw_image)

        
        
        # for i, segment in enumerate(self.segments):
            
            
            # non_segment_kp_indexes = [i for i in range(len(keypoint_locs)) if i not in segment_kp_indexes]
            # new_training_labels[segment_kp_indexes] = 1
            
            # draw_image = draw_points(draw_image, keypoint_locs[non_segment_kp_indexes], color=(0, 0, 255))
            # draw_points(draw_image, keypoint_locs[segment_kp_indexes], color=(255, 0, 0), show=True)
            
            # sides = segment.side_segments(distance=side_distance)
            

            # print(white_color_model.hsv_variances())
            # print(secure_colors[white_side])
            # image_pixels = image.reshape(-1, 3)
            # white_innovation = white_color_model.color_innovation(image_pixels)
            # # white_innovation_image = np.zeros(grayscale_image.shape, dtype=np.float32)
            # white_innovation_image = white_innovation.reshape(grayscale_image.shape)
            # # plt.imshow((1 - white_innovation_image / np.max(white_innovation_image) * 255 ))
            # imshow((white_innovation_image < 3).astype(int) * 255)
            # hsv_variances = white_color_model.hsv_variances()
            # print('hsv_variances:', hsv_variances, np.max(hsv_variances))
        
    def plot(self, plot_segment_process=False, plot_points=True, thickness=10, point_radius=3, return_image=False):
        draw_image = self.image.copy()
        draw_image = draw_points(draw_image, self.hand_points, (0, 255, 255), radius=point_radius, return_image=True)
        for i, candidate in enumerate(self.main_segment_candidates):
            if i == self.lm.selected_line:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            candidate.segment.plot(draw_image, color, thickness)
        if plot_points:
            draw_points(draw_image, self.lm.points, radius=point_radius)
        imshow(draw_image)
        if return_image:
            return draw_image
    
    def plot_segment_process(self):
        for candidate in self.main_segment_candidates:
            plt.figure()
            candidate.plot_process()
