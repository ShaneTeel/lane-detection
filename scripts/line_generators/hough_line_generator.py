import cv2
import numpy as np
import math
import streamlit as st

class HoughLineGenerator():
    '''Test'''
    _DEFAULT_CONFIG = {
                'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
                'canny': {'canny_low': 50, 'canny_high': 100, 'blur_first': False},
                'hough': {'rho': 1, 'theta': np.pi / 180, 'thresh': 50, 'min_length': 10, 'max_gap': 20}
            }
    
    def __init__(self, roi, configs):
        if configs is None:
            configs = self._DEFAULT_CONFIG
        
        if roi is not None:
            roi = np.array([[pt for pt in roi]])
        else:
            roi = self._POLYGON
        self.in_range_params = configs['in_range']
        self.canny_params = configs['canny']
        self.hough_params = configs['hough']
        self.composite_params = configs['composite']
        self.roi = roi
        self._validate_configs() # Validate configs

    def _validate_configs(self):
        attributes = [self.in_range_params, self.canny_params, self.hough_params]
        for i, step in enumerate(self._DEFAULT_CONFIG.keys()):
            parameters = [val for val in self._DEFAULT_CONFIG[step]]
            for param in parameters:
                    if param not in attributes[i]:
                        raise ValueError(f"Missing required threshold parameter: {param}")


        ####################################################
        # ADD VALIDATION FOR PARAMETER VALUE TYPE / RANGES #
        ####################################################

    def fit(self, edge_map):
        '''ADD LATER'''
        hough, lines = self._fit_lines(edge_map, **self.hough_params)
        if lines is None:
            return hough, edge_map
        lanes = self._gen_lane_lines(lines, self.roi)
        return edge_map, lanes 

    def _fit_lines(self, frame, edge_map, rho, theta, thresh, min_length, max_gap):
        hough = np.zeros((edge_map.shape[0], edge_map.shape[1], 3), dtype=np.uint8)
        lines = cv2.HoughLinesP(edge_map, rho, theta, thresh, min_length, max_gap)
        if lines is None:
            return hough, None
        self._draw_lines(hough, lines, color=(255, 255, 255))
        return hough, lines
    
    def _gen_lane_lines(self, lines, poly):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            lanes_classified = self._classify_lines(lines)
            return self._gen_line_of_best_fit(lanes_classified, poly)
    
    def _create_composite(self, frame, lines, poly, stroke, stroke_color, fill, fill_color):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            lanes_classified = self._classify_lines(lines)
            lane_lines = self._gen_line_of_best_fit(lanes_classified, poly)

            canvas = np.zeros([frame.shape[0], frame.shape[1], 3], dtype=np.uint8)
            self._draw_stroke_fill(canvas, lane_lines, stroke, stroke_color, fill, fill_color)

            img = cv2.addWeighted(frame, 0.8, canvas, 0.3, 0.0)
            return img
    
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

    def _draw_stroke_fill(self, img, lines, stroke, stroke_color, fill, fill_color):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            if stroke:
                self._draw_lines(img, [lines[0]], stroke_color, 10)
                self._draw_lines(img, [lines[1]], stroke_color, 10)
            if fill:
                self._draw_fill(img, lines, fill_color)

    def _draw_fill(self, img, lines, color):
        points = np.array([[*[[x1, y1] for x1, y1, _, _ in lines[0]],
                            *[[x2, y2] for _, _, x2, y2 in lines[0]],
                            *[[x2, y2] for _, _, x2, y2 in lines[1]],
                            *[[x1, y1] for x1, y1, _, _ in lines[1]]]], dtype='int32')
        cv2.fillPoly(img=img, pts=points, color=color)

    def _draw_lines(self, img, lines, color, thickness=1):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)