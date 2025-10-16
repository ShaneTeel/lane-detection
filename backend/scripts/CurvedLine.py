import cv2
import numpy as np
import math
import copy

class CannyRANSAC():
    '''Test'''
    _POLYGON = np.array([[[100, 540], 
                          [900, 540], 
                          [515, 320], 
                          [450, 320]]])
    _DEFAULT_CONFIG = {
                'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
                'canny': {'canny_low': 50, 'canny_high': 100, 'blur_first': True},
                'fit': {'n_iter': 100, 'degree': 2, 'threshold': 50, 'min_inliers': 0.6, 'factor': 0.1},
                'composite': {'stroke': True, "stroke_color": (0, 0, 255), 'fill': True, "fill_color": (0, 255, 0)}
            }

    def __init__(self, roi = None, configs = None):
        if configs is None:
            self.configs = self._DEFAULT_CONFIG
        else:
            self.configs = copy.deepcopy(self._DEFAULT_CONFIG)
            self._merge_configs(configs)
        

        self.in_range_params = self.configs['in_range']
        self.canny_params = self.configs['canny']
        self.fit_params = self.configs['fit']
        self.composite_params = self.configs['composite']
        self.prev_points = {"left": None, "right": None}
        
        self._roi_validation(roi)
        self._hex_to_bgr('fill_color')
        self._hex_to_bgr('stroke_color')

    def _merge_configs(self, user_configs):
        def _recursive_update(default_dict, new_dict):
            for key, val in new_dict.items():
                if isinstance(val, dict) and key in default_dict and isinstance(default_dict[key], dict):
                    _recursive_update(default_dict[key], val)
                else:
                    default_dict[key] = val

        _recursive_update(self.configs, user_configs)
        self._validate_configs()

    def _validate_configs(self):
        attributes = [self.in_range_params, self.canny_params, self.fit_params, self.composite_params]
        for i, step in enumerate(self._DEFAULT_CONFIG.keys()):
            parameters = [val for val in self._DEFAULT_CONFIG[step]]
            for param in parameters:
                    if param not in attributes[i]:
                        raise ValueError(f"Missing required threshold parameter: {param}")

    def _roi_validation(self, roi):
        if roi is None:
            self.roi = self._POLYGON
        else:
            expected_shape = (1, 4, 2)
            if roi.shape == expected_shape:
                self.roi = roi
            else:
                try:
                    self.roi = roi.reshape(1, 4, 2)
                except Exception as e:
                    assert AssertionError(e)

    def _hex_to_bgr(self, key):
        hex_color = self.composite_params.get(key)
        if isinstance(hex_color, tuple):
            return
        elif hex_color.startswith("#"):
            hex_color = hex_color[1:]
        
        if len(hex_color) != 6:
            raise ValueError("Invalid hex color code format. Expected six characters (remove alpha channel)")
        
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        self.composite_params[key] = (b, g, r)

    def run(self, frame):
        '''ADD LATER'''
        threshold = self._threshold_lane_lines(frame, **self.in_range_params)
        roi = self._select_ROI(threshold, self.roi)
        edge_map = self._detect_edges(roi, **self.canny_params)
        lane_lines = self._fit_lane_lines(edge_map, **self.fit_params)
        composite = self._create_composite(frame, lane_lines, **self.composite_params)
        return threshold, edge_map, composite

    def _threshold_lane_lines(self, frame, lower_bounds, upper_bounds):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.inRange(gray, lower_bounds, upper_bounds)
        return thresh

    def _select_ROI(self, thresh_img, poly):
        mask = np.zeros_like(thresh_img)
        if len(thresh_img.shape) > 2:
            num_channels = thresh_img.shape[2]
            roi_color = (255,) * num_channels
        else:
            roi_color = 255
        cv2.fillPoly(img=mask, pts=poly, color=roi_color)
        roi = cv2.bitwise_and(src1=thresh_img, src2=mask)
        return roi

    def _detect_edges(self, roi, canny_low, canny_high, blur_first:bool=True):
        if blur_first == True:
            img = cv2.GaussianBlur(roi, (3, 3), 0)
            img = cv2.Canny(img, canny_low, canny_high)
        else:
            img = cv2.Canny(roi, canny_low, canny_high)
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def _fit_lane_lines(self, edge_map, n_iter, degree, threshold, min_inliers, factor):
        if edge_map is None:
            raise ValueError("Error: Argument passed for edge map is none")
        else:
            pts = self._point_extraction(edge_map)
            pts_split = self._point_splitting(pts)
            fit = self._ransac_polyfit(pts_split, n_iter, degree, threshold, min_inliers, factor)
            return fit

    def _create_composite(self, frame, fit, stroke, stroke_color, fill, fill_color):
        if fit is None:
            raise ValueError("Error: argument passed for fit is none.")
        else:
            canvas = np.zeros([frame.shape[0], frame.shape[1], 3], dtype=np.uint8)
            self._draw_curve_stroke_fill(canvas, fit, stroke, stroke_color, fill, fill_color)

            composite = cv2.addWeighted(frame, 0.8, canvas, 0.3, 0.0)

            return composite

    def _point_extraction(self, edge_map):
        edge_pts = np.where(edge_map != 0)
        return np.column_stack((edge_pts[1], edge_pts[0]))
    
    def _point_splitting(self, pts):
        if len(pts) == 0:
            return [np.array([]), np.array([])]

        x_mid = self.roi[:, :, 0].mean()

        left = pts[pts[:, 0] < x_mid]
        right = pts[pts[:, 0] >= x_mid]

        return [left, right]
    
    def _ransac_polyfit(self, lanes, n_iter, degree, threshold, min_inliers, factor):
        fit = []

        for i, lane in enumerate(lanes):
            # X, y, and direction variable creation
            direction = "left" if i == 0 else "right"
            X = lane[:, 0]
            y = lane[:, 1]

            if len(lane) < degree + 1:
                print(f"WARNING: {direction} lane does not have enough points to perform fit. Skipping lane of length {len(lane)}.")
                continue

            # Get best coeffs
            best_coeffs = self._best_coeffs(X, y, n_iter, degree, threshold, min_inliers)

            # Generate Curved Lines
            fit.append(self._gen_smoothed_curved_lines(best_coeffs, direction, factor))

        return fit
    
    def _best_coeffs(self, X, y, n_iter:int=100, degree:int=2, threshold:int=20, min_inliers:float=0.6):
        # Normalize values for polyfit
        X, y, threshold = self._min_max_scaler(X, y, threshold)

        # Create best coeffs check variables
        poly_size = degree + 1
        best_inliers = None
        best_inlier_count = 0
        best_coeffs = None
        n_points = len(X)
        sample_size = min(max(degree+3, 5), n_points)
        
        for _ in range(n_iter):
            # Random sampling
            sample_idx = np.random.choice(n_points, size=sample_size, replace=False)
            sample_X = X[sample_idx]
            sample_y = y[sample_idx]

            # Fit polynomial to samples
            try:
                coeffs = np.polyfit(sample_X, sample_y, degree)

                if not isinstance(coeffs, np.ndarray) or len(coeffs) != poly_size:
                    continue

            except (np.linalg.LinAlgError, TypeError, ValueError):
                continue

            # Evaluate fit on all points
            y_pred = np.polyval(coeffs, X)
            errors = np.abs(y - y_pred)

            # Count inliers (points close to fit)
            inliers = errors < threshold
            inlier_count = np.sum(inliers)

            # Best coeffs check
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_coeffs = coeffs

        # Best coeffs vs normal polyfit check
        if best_inliers is not None and best_inlier_count >= min_inliers * n_points:
            inlier_X = X[best_inliers]
            inlier_y = y[best_inliers]
            if len(inlier_X) >= poly_size:
                try:
                    ransac_coeffs = np.polyfit(inlier_X, inlier_y, degree)
                    if isinstance(ransac_coeffs, np.ndarray) and len(ransac_coeffs) == poly_size:
                        return ransac_coeffs
                except (np.linalg.LinAlgError, TypeError, ValueError):
                    pass
                else:
                    return best_coeffs # Return best coeffs without refit

        # FAIL SAFE #1: If leading coefficient is a negative value, fit all data
        if degree == 2 and best_coeffs is not None:
            if best_coeffs[0] < 0:
                print(f"WARNING: Suspecious parabola a = {best_coeffs[0]}")
                return np.polyfit(X, y, degree)

        # FAIL SAFE #2: Fit all data
        try:
            last_resort = np.polyfit(X, y, degree)
            if isinstance(last_resort, np.ndarray) and len(last_resort) == poly_size:
                return last_resort
        except:
            pass
        
    def _min_max_scaler(self, X, y, threshold):
        targets = [X, y]
        params = []
        for i in range(len(targets)):
            val = targets[i]
            min, max = val.min(), val.max()
            params.append([min, max])
            targets[i] = (val - min) / (max - min) if max > min else val
            if i == 1:
                targets.append(threshold / (max - min) if max > min else threshold)

        self.scale_params = [val for sub in params for val in sub]

        return targets
    
    def _gen_smoothed_curved_lines(self, best_coeffs, direction, factor:float = 0.3):
        # Gen curve in scaled sapce
        x_scaled = np.linspace(0, 1, 100)
        print(best_coeffs)
        y_scaled = np.polyval(best_coeffs, x_scaled)

        # Return scaled values to original coordinate space
        X, y = self._anti_scaler(x_scaled, y_scaled)
        points = np.array([X, y], dtype=np.int32).T

        # Smooth points if able
        if self.prev_points.get(direction) is not None:
            points = (factor * points + (1 - factor) * self.prev_points.get(direction)).astype(np.int32)
        
        # Assign points to persistent variable for next frame
        self.prev_points[direction] = points

        return points.reshape((-1, 1, 2))
    
    def _anti_scaler(self, X_scaled, y_scaled):
        X_min, X_max, y_min, y_max = self.scale_params

        X = X_scaled * (X_max - X_min) + X_min
        y = y_scaled * (y_max - y_min) + y_min
        return X, y

    def _draw_curve_stroke_fill(self, canvas, fit, stroke:bool=True, stroke_color:tuple=(0, 0, 255), fill:bool=True, fill_color:tuple=(0, 255, 0)):
        if fit is None:
            raise ValueError("Error: argument passed fCannyKMeansor lines contains no lines.")
        else:
            if stroke:
                self._draw_curved_lines(canvas, [fit[0]], stroke_color, 10)
                self._draw_curved_lines(canvas, [fit[1]], stroke_color, 10)
            if fill:
                self._draw_fill(canvas, fit, fill_color)

    def _draw_fill(self, frame, fit, color:tuple=(0, 255, 0)):
        if fit is None:
            raise ValueError("Lines are None")
            
        poly = np.concatenate(fit, dtype=np.int32)
        cv2.fillPoly(img=frame, pts=[poly], color=color)

    def _draw_curved_lines(self, frame, points, color:tuple=(0, 0, 255), thickness:int=1):
        if points is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            cv2.polylines(frame, points, isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    
    def _gen_line_of_best_fit(self, lanes, poly):
        yMin = min(poly[0][0][1], poly[0][-1][1])
        yMax = max(poly[0][0][1], poly[0][-1][1])

        for lateral in lanes.keys():
            lanes[lateral]['mAvg'] = self._calc_avg(lanes[lateral]['m'])
            lanes[lateral]['bAvg'] = self._calc_avg(lanes[lateral]['b'])

            if math.isinf(lanes[lateral]['mAvg']) or math.isinf(lanes[lateral]['bAvg']):
                raise ValueError("Error: Infinite float.")
            else:
                try:
                    xMin = int((yMin - lanes[lateral]['bAvg']) // lanes[lateral]['mAvg'])
                    xMax = int((yMax - lanes[lateral]['bAvg']) // lanes[lateral]['mAvg'])
                    lanes[lateral]['line'].append([xMin, yMin, xMax, yMax])
                except Exception as e:
                    raise ValueError(f"Error: {e}")
                
        return [lanes['left']['line'], lanes['right']['line']]

if __name__ == "__main__":

    cap = cv2.VideoCapture("media/lane1-straight.mp4")

    if not cap.isOpened():
        print("Error: Could not open video.")

    processor = CannyRANSAC()
    pause = False

    while True and not pause:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            thresh, edge, composite = processor.run(frame)
            thresh = cv2.merge([thresh, thresh, thresh])
            edge = cv2.merge([edge, edge, edge])
            top = np.hstack([frame, thresh])
            bottom = np.hstack([edge, composite])
            combined = np.vstack([top, bottom])

            cv2.imshow("test", combined)
            key = cv2.waitKey(1)
            if key == 27 or key == 32:
                break
    
    cv2.destroyAllWindows()