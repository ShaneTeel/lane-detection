import cv2
import numpy as np
import math
import streamlit as st

class HoughLineGenerator():
    '''Test'''
    
    def __init__(self, roi, rho, theta, thresh, min_length, max_gap):
        self.rho = rho
        self.theta = theta
        self.thresh = thresh
        self.min_length = min_length
        self.max_gap = max_gap
        self.roi = roi

    def fit(self, edge_map):
        '''ADD LATER'''
        lines = self._fit_lines(edge_map, self.rho, self.theta, self.thresh, self.min_length, self.max_gap)
        if lines is None:
            return None
        lanes = self._gen_lane_lines(lines, self.roi)
        return lanes 

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
        yMin = min(poly[0][0][1], poly[0][-1][1])
        yMax = max(poly[0][0][1], poly[0][-1][1])

        for lateral in lanes.keys():
            lanes[lateral]['mAvg'] = self._calc_avg(lanes[lateral]['m'])
            lanes[lateral]['bAvg'] = self._calc_avg(lanes[lateral]['b'])

            if math.isinf(lanes[lateral]['mAvg']) or math.isinf(lanes[lateral]['bAvg']):
                raise ValueError("Error: Infinite float.")
            
            if abs(lanes[lateral]['mAvg']) < 1e-6:
                continue
            
            xMin = int((yMin - lanes[lateral]['bAvg']) // lanes[lateral]['mAvg'])
            xMax = int((yMax - lanes[lateral]['bAvg']) // lanes[lateral]['mAvg'])
            lanes[lateral]['line'].append([xMin, yMin, xMax, yMax])

        return [lanes['left']['line'], lanes['right']['line']]

    def _classify_lines(self, lines):
        lanes = {'left': {'m': [], 'b': [], 'mAvg': 0, 'bAvg': 0, 'line': []},
                'right': {'m': [], 'b': [], 'mAvg': 0, 'bAvg': 0, 'line': []}}
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
    