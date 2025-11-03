import cv2
import numpy as np
from .canny_feature_engineer import CannyEdgeGenerator


class HoughFeatureEngineer():

    _DEFAULT_CONFIGS = {
        "generator": {
            'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
            'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "after"}
        },
        "extractor": {
            'hough': {'rho': 1, 'theta': 1, 'thresh': 50, 'min_length': 10, 'max_gap': 20},
            "filter": {"filter_type": "median", "n_std": 2.0, "weight": 5}
        }
    }

    def __init__(self, x_mid, configs):
        if configs is None:
            gen_configs, ext_configs = self._DEFAULT_CONFIGS["generator"], self._DEFAULT_CONFIGS["extractor"]
        
        else:
            gen_configs, ext_configs = configs["generator"], configs["extractor"]

        self.generator = CannyEdgeGenerator(gen_configs)
        self.extractor = HoughLineGenerator(x_mid, ext_configs)
    
    def generate_features(self, frame):
        thresh, edge_map = self.generator.generate(frame)
        kps = self.extractor.extract(edge_map)
        return thresh, edge_map, kps        

class HoughLineGenerator():

    _DEFAULT_CONFIGS = {
        'hough': {'rho': 1, 'theta': 1, 'thresh': 50, 'min_length': 10, 'max_gap': 20},
        'filter': {"filter_type": "median", "n_std": 2.0, "weight": 5}
        }

    def __init__(self, x_mid, configs):
        if configs is None:
            hough, filter = self._DEFAULT_CONFIGS["hough"], self._DEFAULT_CONFIGS["filter"]
        else:
            hough, filter = configs["hough"], configs["filter"]

        self.x_mid = x_mid
        self.rho = hough['rho']
        self.theta = np.radians(hough["theta"])
        self.thresh = hough['thresh']
        self.min_length = hough['min_length']
        self.max_gap = hough['max_gap']
        self.filter_type = filter['filter_type']
        self.n_std = filter['n_std']

    def extract(self, edge_map):
        lines = self._point_extraction(edge_map, self.rho, self.theta, self.thresh, self.min_length, self.max_gap)
        if lines is None:
            return None
        # filtered = self._center_filter(lines, self.filter_type, self.n_std)
        lanes = self._point_splitting(lines, self.x_mid)
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

    # def _center_filter(self, lanes, filter_type:str =['median', 'mean'], n_std:float=2.0):
    #     # Lane X-Val Filter
        
    #     if lanes is not None:
    #         X1 = lanes[:, :, 0]
    #         X2 = lanes[:, :, 2]
    #         X = np.concatenate([X1, X2], axis=1)
    #         X_center = np.median(X) if filter_type == "median" else np.mean(X)
    #         X_std = np.std(X)

    #         mask = np.abs(X - X_center) < (n_std * X_std)
    #     return lanes[mask[:, 0]]