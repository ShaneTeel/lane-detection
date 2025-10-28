import cv2
import numpy as np

class HoughPLineGenerator():

    _DEFAULT_CONFIGS = {
        'hough': {'rho': 1, 'theta': 1, 'thresh': 50, 'min_length': 10, 'max_gap': 20},
        'filter': {"filter_type": "median", "n_std": 2.0}
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

    def extract_points(self, edge_map):
        lines = self._point_extraction(edge_map, self.rho, self.theta, self.thresh, self.min_length, self.max_gap)
        if lines is None:
            return None
        lanes = self._point_splitting(lines, self.x_mid)
        return self._point_filtering(lanes, self.filter_type, self.n_std)

    def _point_extraction(self, edge_map, rho, theta, thresh, min_length, max_gap):
        lines = cv2.HoughLinesP(edge_map, rho, theta, thresh, min_length, max_gap)
        if lines is None:
            return None
        return lines

    def _point_splitting(self, lines, x_mid):
        left = []
        right = []
        for line in lines:
            start_x = line[:, 0]
            stop_x = line[:, 2]
            if start_x < x_mid and stop_x <= x_mid:
                left.append(line)
            if start_x >= x_mid and stop_x >= x_mid:
                right.append(line)
        return [left, right]
    
    def _point_filtering(self, lanes, filter_type:str =['median', 'mean'], n_std:float=2.0):
        lanes_filtered = []

        # Lane X-Val Filter
        for lane in lanes:
            if lane is not None:
                X1 = lane[:, 0]
                X2 = lane[:, 2]
                X = np.concatenate
                X_center = np.median(X) if filter_type == "median" else np.mean(X)
                X_std = np.std(X)

                mask = np.abs(X - X_center) < (n_std * X_std)
                lanes_filtered.append(lane[mask])
        return lanes_filtered