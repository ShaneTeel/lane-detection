import numpy as np

class RANSACLineGenerator():
    '''Test'''
    
    _VALID_CONFIG_SETUP = {
        'filter': {'filter_type': ['median', 'mean'], 'n_std': np.arange(0.0, 3.1, 0.01).tolist()},
        'polyfit': {'n_iter': list(range(1, 101)), 'degree': [1, 2, 3], 'threshold': list(range(0, 101)), 'min_inliers': np.arange(0.0, 1.00, 0.01), 'weight': list(range(1, 11)), "factor": np.arange(0.0, 1.00, 0.01)},
    }
    _DEFAULT_CONFIG = {
        'filter': {'filter_type': 'median', 'n_std': 2}, 
        'polyfit': {'n_iter': 100, 'degree': 2, 'threshold': 50, 'min_inliers': 0.6, 'weight': 5, 'factor': 0.1}
    }

    def __init__(self, roi, configs:dict = None):
        if configs is None:
            configs = self._DEFAULT_CONFIG, None

        self.roi = roi
        self.filtering_params = configs['filter']
        self.polyfit_params = configs['polyfit']
        self.prev_points = {"left": None, "right": None}

    def fit(self, edge_map):
        if edge_map is None:
            raise ValueError("Error: Argument passed for edge map is none")
        else:
            pts = self._point_extraction(edge_map)
            pts_split = self._point_splitting(pts)
            pts_filtered = self._point_filtering(pts_split, **self.filtering_params)
            fit = self._ransac_polyfit(pts_filtered, **self.polyfit_params)
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

        return [left, right] 

    def _point_filtering(self, lanes, filter_type:str =['median', 'mean'], n_std:float=2.0):
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
    
    def _ransac_polyfit(self, lanes, n_iter, degree, threshold, min_inliers, weight, factor):
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
            best_coeffs = self._best_coeffs(X, y, n_iter, degree, threshold, min_inliers, weight)

            # Generate Curved Lines
            fit.append(self._gen_smoothed_curved_lines(best_coeffs, direction, factor))
        return fit
    
    def _best_coeffs(self, X, y, n_iter:int=100, degree:int=2, threshold:int=20, min_inliers:float=0.6, weight:int = 5):
        # Normalize values for polyfit
        X, y, threshold = self._min_max_scaler(X, y, threshold)
        self.y = y

        # Constrain y to prevent inaccruate line response
        y_min_idx, y_max_idx = np.argmin(y), np.argmax(y)

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

        # Filter points further by max ROI.y
        points = points[points[:, 1] >= min(self.roi[:, :, 1][0])]
        return points.reshape((-1, 1, 2))
    
    def _inverse_scaler(self, X_scaled, y_scaled):
        X_min, X_max, y_min, y_max = self.scale_params

        X = X_scaled * (X_max - X_min) + X_min
        y = y_scaled * (y_max - y_min) + y_min
        return X, y