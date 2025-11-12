import cv2
import numpy as np
from lane_detection.preprocessing.canny_feature_engineer import CannyEdgeGenerator
from lane_detection.preprocessing.config_manager import ConfigManager


class HoughFeatureEngineer():

    _VALID_SETUP = {
        "generator": {
            'in_range': {'lower_bounds': [0, 255], 'upper_bounds': [0, 255]},
            'canny': {'weak_edge': [0, 301], 'sure_edge': [0, 301], 'blur_ksize': [3, 15], "blur_order": ["before", "after"]}
            },
        "extractor": {
            "hough": {'rho': [1, 1000], 'theta': [1, 180], 'thresh': [1, 500], 'min_length': [1, 1000], 'max_gap': [1, 1000]},
            "filter": {"filter_type": ["median", "mean"], "n_std": [0.1, 5.0], "weight": [0, 100]}
        }
    }

    _DEFAULT_CONFIGS = {
        "generator": {
            'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
            'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "after"}
        },
        "extractor": {
            'hough': {'rho': 1, 'theta': 1, 'thresh': 50, 'min_length': 15, 'max_gap': 10},
            "filter": {"filter_type": "median", "n_std": 2.0, "weight": 5}
        }
    }

    def __init__(self, configs:dict = None):
        if configs is None:
            gen_configs, ext_configs = self._DEFAULT_CONFIGS['generator'], self._DEFAULT_CONFIGS["extractor"]
        else:
            final = get_configs(configs, self._DEFAULT_CONFIGS, self._VALID_SETUP)
            gen_configs, ext_configs = final["generator"], final["extractor"]

        self.generator = CannyEdgeGenerator(gen_configs)
        self.extractor = HoughLineGenerator(ext_configs)
    
    def transform(self, frame, x_mid):
        thresh, edge_map = self.generator.generate(frame)
        kps = self.extractor.extract(edge_map, x_mid)
        return thresh, edge_map, kps   

class HoughLineGenerator():
    
    _VALID_SETUP = {
        "hough": {'rho': [1, 1000], 'theta': [1, 180], 'thresh': [1, 500], 'min_length': [1, 1000], 'max_gap': [1, 1000]}
    }

    _DEFAULT_CONFIGS = {
        "hough": {'rho': 1, 'theta': 1, 'thresh': 50, 'min_length': 15, 'max_gap': 10}
    }

    def __init__(self, configs:dict=None):
        if configs is None:
            hough = self._DEFAULT_CONFIGS["hough"]
        else:
            hough = get_configs(configs, self._DEFAULT_CONFIGS, self._VALID_SETUP)["hough"]

        self.rho = hough['rho']
        self.theta = np.radians(hough["theta"])
        self.thresh = hough['thresh']
        self.min_length = hough['min_length']
        self.max_gap = hough['max_gap']

    def extract(self, edge_map, x_mid):
        lines = self._point_extraction(edge_map, self.rho, self.theta, self.thresh, self.min_length, self.max_gap)
        if lines is None:
            return None
        lanes = self._point_splitting(lines, x_mid)
        return lanes

    def _point_extraction(self, edge_map, rho, theta, thresh, min_length, max_gap):
        lines = cv2.HoughLinesP(edge_map, rho, theta, thresh, min_length, max_gap)
        if lines is None:
            return None
        return lines

    def _point_splitting(self, lines, x_mid):
        left = []
        right = []
        for line in lines:
            X1, y1, X2, y2 = line.flatten()
            if X1 < x_mid and X2 <= x_mid:
                left.append([X1, y1])
                left.append([X2, y2])
            if X1 >= x_mid and X2 >= x_mid:
                right.append([X1, y1])
                right.append([X2, y2])
        return [np.array(left), np.array(right)]
    
def get_configs(user_configs, default_configs, valid_config_setup):
    config_mngr = ConfigManager(user_configs, default_configs, valid_config_setup)
    final = config_mngr.manage()
    return final