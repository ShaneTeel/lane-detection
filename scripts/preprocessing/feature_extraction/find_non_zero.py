import numpy as np
from typing import Literal

class FindNonZero():
            
    _DEFAULT_CONFIGS = {
        'filter': {"filter_type": "median", "n_std": 2.0}
    }

    def __init__(self, x_mid, filter_type, n_std):
        self.x_mid = x_mid
        self.filter_type = filter_type
        self.n_std = n_std

    def extract_points(self, edge_map):
        pts = self._point_extraction(edge_map)
        classified = self._point_splitting(pts, self.x_mid)
        return self._point_filtering(classified, self.filter_type, self.n_std)

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

    def _point_filtering(self, lanes, filter_type:Literal['median', 'mean']="median", n_std:float=2.0):
        lanes_filtered = []

        # Lane X-Val Filter
        for lane in lanes:
            if lane is not None:
                X = lane[:, 0]
                X_center = np.median(X) if filter_type == "median" else np.mean(X)
                X_std = np.std(X)

                mask = np.abs(X - X_center) < (n_std * X_std)
                lanes_filtered.append(lane[mask])

        return lanes_filtered