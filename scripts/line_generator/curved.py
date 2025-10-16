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
        self.prev_points = {"left": None, "right": None}
        
        self._roi_validation(roi)

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
        attributes = [self.in_range_params, self.canny_params, self.fit_params]
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

    def run(self, frame):
        '''ADD LATER'''
        threshold = self._threshold_lane_lines(frame, **self.in_range_params)
        roi = self._select_ROI(threshold, self.roi)
        edge_map = self._detect_edges(roi, **self.canny_params)
        lane_lines = self._fit_lane_lines(edge_map, **self.fit_params)
        return threshold, edge_map, lane_lines
    
    def _threshold_lane_lines(self, frame, lower_bounds, upper_bounds):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.inRange(img, lower_bounds, upper_bounds)
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

    def _point_extraction(self, edge_map):
        edge_pts = np.where(edge_map != 0)
        return np.column_stack((edge_pts[1], edge_pts[0]))
    
    def _point_splitting(self, pts):
        if len(pts) == 0:
            return [np.array([]), np.array([])]

        x_mid = self.roi[:, :, 0].mean()

        left = pts[pts[:, 0] < x_mid]
        right = pts[pts[:, 0] >= x_mid]

        lanes_raw = [left, right]
        lanes_filtered = []

        # Lane X-Val Filter
        for lane in lanes_raw:
            if lane is not None:
                X = lane[:, 0]
                X_median = np.median(X)
                X_std = np.std(X)

                mask = np.abs(X - X_median) < (2 * X_std)
                lanes_filtered.append(lane[mask])

        return lanes_filtered
    
    def _ransac_polyfit(self, lanes, n_iter, degree, threshold, min_inliers, factor):
        fit = []

        for i, lane in enumerate(lanes):
            # Direction variable
            direction = "left" if i == 0 else "right"
            
            if len(lane) < degree + 1:
                print(f"WARNING: {direction} lane does not have enough points to perform fit. Skipping lane of length {len(lane)}.")
                continue
            
            # X-Value Filter 
            X = lane[:, 0]
            y = lane[:, 1]

            y_range = y.max() - y.min()
            top_third = lane[lane[:, 1] < (y.min() + y_range / 3)]

            if len(top_third) < 5: 
                print(f"{direction}: sparse top, using degree = 1")
                degree = 1

            # Get best coeffs
            best_coeffs = self._best_coeffs(X, y, n_iter, degree, threshold, min_inliers)

            # Generate Curved Lines
            fit.append(self._gen_smoothed_curved_lines(best_coeffs, direction, factor))

        return fit
    
    def _best_coeffs(self, X, y, n_iter:int=100, degree:int=2, threshold:int=20, min_inliers:float=0.6):
        # Normalize values for polyfit
        X, y, threshold = self._min_max_scaler(X, y, threshold)
        self.y = y

        # Constrain y to prevent inaccruate line response
        y_min_idx, y_max_idx = np.argmin(y), np.argmax(y)

        weight = 5
        X_constrained = np.concatenate([X] + [X[[y_min_idx, y_max_idx]]] * weight)
        y_constrained = np.concatenate([y] + [y[[y_min_idx, y_max_idx]]] * weight)

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
            sample_X = X_constrained[sample_idx]
            sample_y = y_constrained[sample_idx]

            # Fit polynomial to samples
            try:
                coeffs = np.polyfit(sample_X, sample_y, degree)

                if not isinstance(coeffs, np.ndarray) or len(coeffs) != poly_size:
                    continue

            except (np.linalg.LinAlgError, TypeError, ValueError):
                continue

            # Evaluate fit on all points
            y_pred = np.polyval(coeffs, X)
            errors = np.abs(sum(y - y_pred) ** 2)

            # Count inliers (points close to fit)
            inliers = errors < threshold
            inlier_count = np.sum(inliers)

            # Best coeffs check
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_coeffs = coeffs

        # Leading Coefficient check: If leading coefficient is a negative value, fit all data
        if degree == 2 and best_coeffs is not None:
            if best_coeffs[0] < 0:
                # print(f"WARNING: Suspecious parabola a = {best_coeffs[0]}")
                return np.polyfit(X_constrained, y_constrained, degree)

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

        # FAIL SAFE: Fit all data
        try:
            last_resort = np.polyfit(X_constrained, y_constrained, degree)
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
        # Gen curve in scaled space

        x_scaled = np.linspace(0, 1, 100)
        y_scaled = np.polyval(best_coeffs, x_scaled)

        # Return scaled values to original coordinate space
        X, y = self._inverse_scaler(x_scaled, y_scaled)
        points = np.array([X, y], dtype=np.int32).T

        # Smooth points if able
        if self.prev_points.get(direction) is not None:
            points = (factor * points + (1 - factor) * self.prev_points.get(direction)).astype(np.int32)
        
        # Assign points to persistent variable for next frame
        self.prev_points[direction] = points

        return points.reshape((-1, 1, 2))
    
    def _inverse_scaler(self, X_scaled, y_scaled):
        X_min, X_max, y_min, y_max = self.scale_params

        X = X_scaled * (X_max - X_min) + X_min
        y = y_scaled * (y_max - y_min) + y_min
        return X, y

if __name__ == "__main__":

    # cap = cv2.VideoCapture("media/test_img1.jpg")
    
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