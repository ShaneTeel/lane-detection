import numpy as np
from typing import Literal

class CannyFeatureExtractor:

    _DEFAULT_CONFIGS = {
        'extractor': {"filter_type": "median", "n_std": 2.0, "weight": 5}
    }
    def __init__(self, x_mid, configs:dict = None):
        if configs is None:
            configs = self._DEFAULT_CONFIGS["extractor"]

        self.x_mid = x_mid
        self.filter_type = configs["filter_type"]
        self.n_std = configs["n_std"]
        self.weight = configs["weight"]

    def extract_points(self, edge_map):
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
                lane = self._point_filtering(lane, filter_type, n_std)
                lane = self._point_replication(lane, weight)
                resampled.append(lane)

        return resampled

    def _point_filtering(self, lane, filter_type:Literal["median", "mean"]="median", n_std:float=2.0):
        X = lane[:, 0]

        X_center = np.median(X) if filter_type == "median" else np.mean(X)
        X_std = np.std(X)
        X_mask = np.abs(X - X_center) < (n_std * X_std)
        return lane[X_mask]


    def _point_replication(self, lane, weight:int=5):
        X = lane[:, 0]
        y = lane[:, 1]
        y_min_idx, y_max_idx = np.argmin(y), np.argmax(y)
        X = np.concatenate([X] + [X[[y_min_idx, y_max_idx]]] * weight)
        y = np.concatenate([y] + [y[[y_min_idx, y_max_idx]]] * weight)
    
        return np.column_stack([X, y])