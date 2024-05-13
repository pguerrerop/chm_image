from lib.sm_utils.lib.list_and_dict import dict_elem_with_default
from lib.sm_image.color_model import ColorModel
from lib.sm_image.line_segment import get_impulsive_local_minima
from scipy.ndimage import binary_dilation
from lib.sm_utils.lib.utils import value_with_default
import numpy as np
from matplotlib import pyplot as plt
from lib.sm_utils.lib.plot import shade_segment_list
from lib.sm_image.line_segment import get_true_segments, LineSegment


class SidedColorModels:
    def __init__(self, image, side_segments):
        side_int_means = [s.int_mean for s in side_segments]
        non_hand_colors = [s.non_hand_colors for s in side_segments]
        self.white_side_index = np.argmax(side_int_means)
        self.black_side_index = 1 - self.white_side_index
        self.color_models = []
        max_intensities = []
        min_intensities = []
        for i in [self.white_side_index, self.black_side_index]:    
            color_model = ColorModel(image)
            color_model.set_colors(non_hand_colors[i])
            # color_model = color_model.pruned_to_limit_hsv_variance(30, 20, None)
            self.color_models.append(color_model)
            intensity_values = color_model.intensity_values()
            max_intensities.append(np.max(intensity_values))
            min_intensities.append(np.min(intensity_values))
        self.max_intensity = np.max(max_intensities)
        self.min_intensity = np.min(min_intensities)

    def sorted_indices(self):
        return [self.white_side_index, self.black_side_index]

    def get_color_histograms(self, normalize=True):
        histograms = []
        for color_model in self.color_models:
            histograms.append(color_model.intensity_histogram(n_bins=4, min_value=self.min_intensity, max_value=self.max_intensity,
                                                              normalize=normalize))
        return np.hstack(histograms)
    
    
class DetectionParams:
    def __init__(self, params=None):
        params = value_with_default(params, {})

class KeyboardMainSegmentParams(DetectionParams):
    def __init__(self, params=None):
        super().__init__(params)
        self.max_side_distance = dict_elem_with_default(params, 'max_side_distance', 30)
        self.max_side_image_fraction = dict_elem_with_default(params, 'max_side_image_fraction', 25)
        self.kp_filter_distance = dict_elem_with_default(params, 'kp_filter_distance', 10)
        side_params = dict_elem_with_default(params, 'side', {})
        self.side = KeyboardMainSegmentSideParams(side_params)


class KeyboardMainSegmentSideParams(DetectionParams):
    def __init__(self, params=None):
        super().__init__(params)
        self.sampling_distance = dict_elem_with_default(params, 'sampling_distance', 1)
        self.local_minima_diplacement = dict_elem_with_default(params, 'local_minima_diplacement', 10)
        self.dilation_structure_size = dict_elem_with_default(params, 'dilation_structure_size', 21)


class KeyboardMainSegmentCandidateSide:
    def __init__(self, parent, detector, segment):
        self.params = parent.params.side
        self.parent = parent
        self.detector = detector
        self.segment = segment
        self.valid = self.segment.valid
        self.image = parent.image
        self.grayscale_image = parent.grayscale_image
        self.dilation_structure = np.ones(self.params.dilation_structure_size, dtype=bool)
        
    def calculate_minima(self):
        self.minima = get_impulsive_local_minima(self.intensity_values, displacement=self.params.local_minima_diplacement)
        self.minima_indices = np.where(self.minima)[0]

    def calculate_hand_regions(self):
        color_values = self.segment.pixel_values(self.image, sampling_distance=self.params.sampling_distance)
        hand_innovation_thr = self.detector.params.hand_color_innovation_threshold
        self.hand_present = self.detector.hand_cm.color_belonging(color_values, default_value=False, 
                                                                innovation_threshold=hand_innovation_thr)
        # minima_distances = LinearDistancesHandler.from_boolean_vector(minima[~hand_present])
        self.hand_regions = binary_dilation(self.hand_present, structure=self.dilation_structure)
        self.non_hand_colors = color_values[~self.hand_regions]

    def calculate_secure_int_values(self):
        self.restricted_regions = binary_dilation(self.hand_present | self.minima, structure=self.dilation_structure)
        self.secure_int_values = self.intensity_values[~self.restricted_regions]
        self.int_mean = np.mean(self.secure_int_values)
        self.int_max = np.max(self.secure_int_values)
        self.int_min = np.min(self.secure_int_values)
        self.int_range = self.int_max - self.int_min
    
    def calculate_intensity_extreme_jumps(self):
        disp = 5
        high_threshold = self.int_min + .8 * self.int_range
        low_threshold = self.int_min + .2 * self.int_range
        high_intensity = self.intensity_values > high_threshold
        low_intensity = self.intensity_values < low_threshold
        high_jump = high_intensity[disp:] & low_intensity[:-disp]
        low_jump = low_intensity[disp:] & high_intensity[:-disp]
        self.high_jumps = np.where(high_jump)[0]
        self.low_jumps = np.where(low_jump)[0]
    
    def n_extreme_jumps(self):
        return [len(self.high_jumps), len(self.low_jumps)]
        
    def calculate(self):
        if not self.valid:
            return False
        int_values = self.segment.pixel_values(self.grayscale_image, sampling_distance=self.params.sampling_distance)
        self.intensity_values = int_values
        self.calculate_minima()
        self.calculate_hand_regions()
        if self.hand_regions.all():
            return False
        self.calculate_secure_int_values()
        self.calculate_intensity_extreme_jumps()
        return True
    
    def plot_process(self):
        int_values = self.intensity_values
        # if not ss.valid:
        #     self.valid = False
        #     break
        plt.figure()
        plt.plot(self.intensity_values)        
        plt.plot(self.minima.astype(int)*200)
        hand_regions = binary_dilation(self.hand_present, structure=self.dilation_structure)

        hand_segments = get_true_segments(self.hand_present)
        restricted_segments = get_true_segments(self.restricted_regions)
        shade_segment_list(int_values, restricted_segments, color='orange', alpha=0.5)
        hand_restr_segments = get_true_segments(hand_regions)
        shade_segment_list(int_values, hand_restr_segments, color='orange', alpha=0.5)
        shade_segment_list(int_values, hand_segments, color='red', alpha=0.3)

    def plot(self, image):
        self.segment.plot(image, color=(0, 255, 0))

        
class KeyboardMainSegmentCandidate:
    def __init__(self, detector, segment):
        self.detector = detector
        self.segment = segment
        self.image = detector.image
        self.grayscale_image = detector.grayscale_image
        self.params = detector.params.main_segment
        self.image_size = (self.detector.image_height, self.detector.image_width)
        self.kp_indexes = segment.filtered_indices(self.detector.keypoint_locs, self.params.kp_filter_distance)
        max_side_distance = np.min(self.image_size) / self.params.max_side_image_fraction
        self.side_distance = np.min((self.params.max_side_distance, max_side_distance))
        s_segments = segment.side_segments(distance=self.side_distance)
        self.side_segments = [KeyboardMainSegmentCandidateSide(self, detector, s) for s in s_segments]
        self.valid = segment.valid
        self.sided_models = None

    def to_dict(self):
        return {
            'segment': self.segment,
            'valid': self.valid
        }
    
    @staticmethod
    def from_dict(d, detector):
        segment = LineSegment.from_dict(d['segment'])
        return KeyboardMainSegmentCandidate(detector, segment)

    def sort_endpoints(self):
        if self.segment.pt[1][1] > self.segment.pt[0][1]:
            self.segment.revert_endpoints()

    def calculate_side_models(self):        
        for ss in self.side_segments:
            valid = ss.calculate()
            if not valid:
                self.valid = False
                return False
        self.sided_models = SidedColorModels(self.image, self.side_segments)
    # self.lm.plot(self.image.copy(), along_distance_threshold=100, along_distance_ratio_threshold=.06)
    # self.training_labels_list.append(new_training_labels)

    def get_feature_vector(self):
        if not self.valid:
            return None
        color_histograms = self.sided_models.get_color_histograms()
        # if len(self.segments) == 1:
        #     training_histogram_list.append(color_histograms)
        sorted_side_segments = [self.side_segments[i] for i in self.sided_models.sorted_indices()]
        n_minima = [len(ss.minima_indices) for ss in sorted_side_segments]
        n_jumps = np.hstack([ss.n_extreme_jumps() for ss in sorted_side_segments])
        # return color_histograms
        # print(color_histograms.shape, np.shape(n_minima), np.shape(n_jumps))
        return np.hstack((color_histograms, n_minima, n_jumps))
        
    def innovation(self):
        if not self.valid:
            return np.array([np.inf]).reshape(1, -1)
        feature_vector = self.get_feature_vector()
        return self.detector.main_segment_model.innovation(feature_vector.reshape(1, -1))
    
    def score(self):
        if not self.valid:
            return np.array([-np.inf]).reshape(1, -1)
        feature_vector = self.get_feature_vector()
        return self.detector.main_segment_model.score(feature_vector.reshape(1, -1))
    
    def plot(self, image, plot_sides=False, imshow=False):
        self.segment.plot(image, color=(255, 0, 255))
        if plot_sides:
            for ss in self.side_segments:
                ss.plot(image)
        if imshow:
            imshow(image)

    def plot_process(self):
        for ss in self.side_segments:
            ss.plot_process()
            
        # print('side_int_means:', side_int_means)
        # if not self.valid:
        #     return
    # self.lm.plot(self.image.copy(), along_distance_threshold=100, along_distance_ratio_threshold=.06)
    # self.training_labels_list.append(new_training_labels)

        