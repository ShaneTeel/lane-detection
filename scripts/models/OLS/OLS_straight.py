import numpy as np
import math

class StraightLineGenerator:
    '''Test'''
    _DEFAULT_CONFIGS = {'hough': {'rho': 1, 'theta': 1, 'thresh': 50, 'min_length': 10, 'max_gap': 20}}
    
    def __init__(self, y_min, y_max):

        self.y_min = y_min
        self.y_max = y_max
    
    def fit(self, lanes):
        if lanes is None:
            print("WARNING: Argument passed to lines is None. Skipping line")
        
        fit = []
        for lane in lanes:
            m = []
            b = []

            for pt in lane:
                X = lane[0]
                y = lane[1]
                m, b = self._calc_slope_intercept(X, y)
                m.append(m)
                b.append(b)
            
            m_avg = self._calc_avg(m)
            b_avg = self._calc_avg(b)
            start, stop = self._get_best_fit(X, y, m_avg, b_avg)
            fit.append([start, stop])
        return np.array(fit).reshape(-1, 2)

    def _get_best_fit(self, X, y, m_avg, b_avg):
        x1 = int((self.y_max - b_avg) / m_avg)
        x2 = int((self.y_min - b_avg) / m_avg)
        return [[x1, self.y_max], [x2, self.y_min]]
    
    def _calc_slope_intercept(self, X, y):
        if X is None or y is None:
            raise ValueError(f"Error: {X} or {y} == 'NoneType'")
        else:
            
            mean_x = self._calc_avg(X)
            mean_y = self._calc_avg(y)

            numerator = sum((X - mean_x) * (y - mean_y))
            denominator = sum((X - mean_x)**2)
            m = numerator / denominator
            b = mean_y - m * mean_x
            return m, b

    def _calc_avg(self, values):
        if values is None:
            raise ValueError(f"Error: {values} == 'NoneType'")
        else:
            if len(values) > 0:
                n = len(values)
            else:
                n = 1
            return sum(values) / n
        
    def _roi_extraction(self, roi):
        self.x_mid = roi[:, :, 0].mean()
        self.y_min = int(min(roi[0, 0, 1], roi[0, -1, 1]))
        self.y_max = int(max(roi[0, 0, 1], roi[0, -1, 1]))