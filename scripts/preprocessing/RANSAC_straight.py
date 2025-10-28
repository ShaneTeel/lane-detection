import numpy as np

class StraightLineGenerator:
    '''Test'''
    
    def __init__(self, y_min, y_max):

        self.y_min = y_min
        self.y_max = y_max
    
    def fit(self, lanes):
        if lanes is None:
            print("WARNING: Argument passed to lines is None. Skipping line")
        
        fit = []
        for lane in lanes:
            slope = []
            intercept = []
            print(f"DEBUG (PT): {pt}")

            for pt in lane:
                print(f"DEBUG (PT): {pt}")
                X = pt[0]
                y = pt[1]
                m, b = self._calc_slope_intercept(X, y)
                slope.append(m)
                intercept.append(b)
            
            m_avg = self._calc_avg(slope)
            b_avg = self._calc_avg(intercept)
            start, stop = self._get_best_fit(m_avg, b_avg)
            fit.append([start, stop])
        return np.array(fit).reshape(-1, 2)

    def _get_best_fit(self, m_avg, b_avg):
        x1 = int((self.y_max - b_avg) / m_avg)
        x2 = int((self.y_min - b_avg) / m_avg)
        return [[x1, self.y_max], [x2, self.y_min]]
    
    def _calc_slope_intercept(self, X, y):
        if X is None or y is None:
            raise ValueError(f"Error: {X} or {y} == 'NoneType'")
        else:
            print(X)
            
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