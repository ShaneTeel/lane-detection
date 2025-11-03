import cv2
import numpy as np
from typing import Literal

class CannyFeatureEngineer():

    _DEFAULT_CONFIGS = {
            "generator": {
                'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
                'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "after"},
            },
            'extractor': {"filter_type": "median", "n_std": 2.0, "weight": 5}
        }
    
    def __init__(self, x_mid, configs:dict=None):
        if configs is None:
            gen_configs, ext_configs = self._DEFAULT_CONFIGS["generator"], self._DEFAULT_CONFIGS["extractor"]
        
        else:
            gen_configs, ext_configs = configs["generator"], configs["extractor"]

        self.generate = CannyEdgeGenerator(gen_configs)
        self.extract = CannyFeatureExtractor(x_mid, ext_configs)

    def generate_features(self, frame):
        thresh, edge_map = self.generate.generate(frame)
        kps = self.extract.extract(edge_map)
        return thresh, edge_map, kps
    
class CannyEdgeGenerator():
    _DEFAULT_CONFIGS = {
        'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
        'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "after"}
    }

    def __init__(self, configs:dict=None):

        if configs is None:
            configs = self._DEFAULT_CONFIGS
        self.lower_bounds = configs["in_range"].get("lower_bounds")
        self.upper_bounds = configs["in_range"].get("upper_bounds")
        self.weak_edge = configs["canny"].get("weak_edge")
        self.sure_edge = configs["canny"].get("sure_edge")
        self.blur_ksize = configs["canny"].get("blur_ksize")
        self.blur_order = configs["canny"].get("blur_order")
        
    def generate(self, frame):
        thresh = self._threshold(frame, self.lower_bounds, self.upper_bounds)
        edge_map = self._detect_edges(thresh, self.weak_edge, self.sure_edge, self.blur_ksize, self.blur_order)
        return thresh, edge_map
    
    def _threshold(self, frame, lower_bounds, upper_bounds):
        thresh = cv2.inRange(frame, lower_bounds, upper_bounds)
        return thresh

    def _detect_edges(self, roi, weak_edge, sure_edge, blur_ksize, blur_order):
        kernel = (blur_ksize, blur_ksize)
        if blur_order == 'before':
            img = cv2.GaussianBlur(roi, kernel, 0)
            img = cv2.Canny(img, weak_edge, sure_edge)
        else:
            img = cv2.Canny(roi, weak_edge, sure_edge)
            img = cv2.GaussianBlur(img, kernel, 0)
        return img
    
class CannyFeatureExtractor():

    _DEFAULT_CONFIGS = {
        'extractor': {"filter_type": "median", "n_std": 2.0, "weight": 5}
    }
    def __init__(self, x_mid, configs:dict=None):
        if configs is None:
            configs = self._DEFAULT_CONFIGS["extractor"]
           
        self.x_mid = x_mid
        self.filter_type = configs["filter_type"]
        self.n_std = configs["n_std"]
        self.weight = configs["weight"]

    def extract(self, edge_map):
        pts = self._point_extraction(edge_map)
        classified = self._point_splitting(pts, self.x_mid)
        return self._point_resampling(classified, self.filter_type, self.n_std, self.weight)
    
    def _point_extraction(self, edge_map):
        edge_pts = np.where(edge_map != 0)
        if edge_pts is None:
            return np.array([])
        return np.column_stack((edge_pts[1], edge_pts[0]))
    
    def _point_splitting(self, pts, x_mid):
        if len(pts) == 0:
            return [np.array([]), np.array([])]

        left = pts[pts[:, 0] < x_mid]
        right = pts[pts[:, 0] >= x_mid]
        return [left, right]

    def _point_resampling(self, lanes, filter_type, n_std, weight):
        resampled = []

        # Lane X-Val Filter
        for lane in lanes:
            if lane is not None:
                lane = self._X_point_filtering(lane, filter_type, n_std)
                # lane = self._point_replication(lane, weight)
                resampled.append(lane.astype(np.float32))
        return resampled

    def _X_point_filtering(self, lane, filter_type:Literal["median", "mean"]="median", n_std:float=2.0):
        X = lane[:, 0]

        X_center = np.median(X) if filter_type == "median" else np.mean(X)
        X_std = np.std(X)
        X_mask = np.abs(X - X_center) < (n_std * X_std)
        return lane[X_mask]