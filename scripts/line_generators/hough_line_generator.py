import cv2
import numpy as np
import math

class HoughLineGenerator():
    '''Test'''
    _DEFAULT_CONFIGS = {
        'hough': {'rho': 1.0, 'theta': np.radians(1), 'thresh': 50, 'min_length': 10, 'max_gap': 20}
    }
    
    def __init__(self, roi, configs):
        if configs is None:
            configs = self._DEFAULT_CONFIGS["hough"]
        else:
            configs = configs["hough"]

        self.rho = configs["rho"]
        self.theta = configs["theta"]
        self.thresh = configs["thresh"]
        self.min_length = configs["min_length"]
        self.max_gap = configs["max_gap"]
        self.roi = roi

    def fit(self, edge_map):
        '''ADD LATER'''
        lines = self._fit_lines(edge_map, self.rho, self.theta, self.thresh, self.min_length, self.max_gap)
        if lines is None:
            return None
        left, right = self._gen_lane_lines(lines, self.roi)
        return [left, right] 

    def _fit_lines(self, edge_map, rho, theta, thresh, min_length, max_gap):
        lines = cv2.HoughLinesP(edge_map, rho, theta, thresh, min_length, max_gap)
        if lines is None:
            return None
        return lines
    
    def _gen_lane_lines(self, lines, poly):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            lanes_classified = self._classify_lines(lines)
            return self._gen_line_of_best_fit(lanes_classified, poly)
    
    def _gen_line_of_best_fit(self, lanes, poly):
        y_min = int(min(poly[0, 0, 1], poly[0, -1, 1]))
        y_max = int(max(poly[0, 0, 1], poly[0, -1, 1]))

        for lateral in lanes.keys():
            lanes[lateral]['m_avg'] = self._calc_avg(lanes[lateral]['m'])
            lanes[lateral]['b_avg'] = self._calc_avg(lanes[lateral]['b'])

            if math.isinf(lanes[lateral]['m_avg']) or math.isinf(lanes[lateral]['b_avg']):
                raise ValueError("Error: Infinite float.")
            
            if abs(lanes[lateral]['m_avg']) < 1e-6:
                continue
            
            x_min = int((y_min - lanes[lateral]['b_avg']) // lanes[lateral]['m_avg'])
            x_max = int((y_max - lanes[lateral]['b_avg']) // lanes[lateral]['m_avg'])
            lanes[lateral]['line'] = [x_min, y_min, x_max, y_max]

        return lanes['left']['line'], lanes['right']['line']

    def _classify_lines(self, lines):
        lanes = {'left': {'m': [], 'b': [], 'm_avg': 0, 'b_avg': 0, 'line': None},
                'right': {'m': [], 'b': [], 'm_avg': 0, 'b_avg': 0, 'line': None}}
        for line in lines:
            m, b = self._calc_slope_intercept(*line)
            if m is not None:
                if m < 0:
                    lanes['left']['m'].append(m)
                    lanes['left']['b'].append(b)
                elif m > 0:
                    lanes['right']['m'].append(m) 
                    lanes['right']['b'].append(b)
        return lanes

    def _calc_slope_intercept(self, line):
        if line is None:
            raise ValueError(f"Error: {line} == 'NoneType'")
        else:
            x1, y1, x2, y2 = line
            if x1 == x2:
                print(f"Warning: Vertical line detected {line}, skipping")
                return None, None
            
            m = (y1 - y2) / (x1 - x2)
            b = y1 - (m * x1)
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
    