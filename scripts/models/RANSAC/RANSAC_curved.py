import cv2
import numpy as np

class RANSACCurvedLineGenerator():
    '''Test'''
    _DEFAULT_CONFIGS = {
        'polyfit': {'n_iter': 100, 'degree': 2, 'threshold': 50, 'min_inliers': 0.6, 'weight': 5, 'factor': 0.1}
    }
    
    def __init__(self, roi, configs):
        if configs is None:
            configs = self._DEFAULT_CONFIGS["hough"]
        else:
            configs = configs["hough"]

        self.rho = configs["rho"]
        self.theta = np.radians(configs["theta"])
        self.thresh = configs["thresh"]
        self.min_length = configs["min_length"]
        self.max_gap = configs["max_gap"]

        self._roi_extraction(roi)

    def fit(self, edge_map):
        '''ADD LATER'''
        lines = self._hough_lines_p(edge_map, self.rho, self.theta, self.thresh, self.min_length, self.max_gap)
        if lines is None:
            return None
        lines = self._fit_hough_lines(lines)
        return lines 

    def _hough_lines_p(self, edge_map, rho, theta, thresh, min_length, max_gap):
        lines = cv2.HoughLinesP(edge_map, rho, theta, thresh, min_length, max_gap)
        if lines is None:
            return None
        return lines
    
    def _fit_hough_lines(self, lines):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            lines = self._classify_lines(lines)
            fit = self._polyfit(lines, 2)

            return fit
        
    def _classify_lines(self, lines):
        left = []
        right = []
        for line in lines:
            start_x = line[:, 0]
            stop_x = line[:, 2]
            if start_x < self.x_mid and stop_x <= self.x_mid:
                left.append(line)
            if start_x >= self.x_mid and stop_x >= self.x_mid:
                right.append(line)
        return [np.array(left).reshape(-1, 2), np.array(right).reshape(-1, 2)]
        
    def _polyfit(self, lines, degree):
        fit = []
        for i, lane in enumerate(lines):
            direction = "left" if i == 0 else "right"
            if len(lane) < degree + 1:
                print(f"WARNING: {direction} lane does not have enough points to perform fit. Skipping {direction} line.")
                continue
            X = lane[:, 0]
            y = lane[:, 1]

            X_scaled, y_scaled = self._min_max_scaler(X, y)

            coeffs = np.polyfit(X_scaled, y_scaled, 2)
            X_scaled = np.linspace(0, 1, 100)
            y_scaled = np.polyval(coeffs, X_scaled)

            X, y = self._inverse_scaler(X_scaled, y_scaled)
            points = np.array([X, y], dtype=np.int32).T
            fit.append(points.reshape((-1, 1, 2)))
        return fit

    def _min_max_scaler(self, X, y):
        targets = [X, y]
        params = []
        for i in range(len(targets)):
            val = targets[i]
            min, max = val.min(), val.max()
            params.append([min, max])
            targets[i] = (val - min) / (max - min) if max > min else val

        self.scale_params = [val for sub in params for val in sub]

        return targets
    
    def _inverse_scaler(self, X_scaled, y_scaled):
        X_min, X_max, y_min, y_max = self.scale_params

        X = X_scaled * (X_max - X_min) + X_min
        y = y_scaled * (y_max - y_min) + y_min
        return X, y

    def _roi_extraction(self, roi):
        self.x_mid = roi[:, :, 0].mean()
        self.y_min = int(min(roi[0, 0, 1], roi[0, -1, 1]))
        self.y_max = int(max(roi[0, 0, 1], roi[0, -1, 1]))