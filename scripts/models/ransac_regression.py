import numpy as np
from .metrics import RegressionJudge

class RANSACRegression():
    '''Test'''

    _DEFAULT_CONFIG = {
        "estimator": {"method": "OLS", "degree": 2, "factor":0.6, "n_iter": None, "min_inliers": None, "threshold": None}
    }

    def __init__(self, configs:dict = None):
        if configs is None:
            configs = self._DEFAULT_CONFIG["estimator"]

        self.method = configs["method"]
        self.degree = configs['degree']
        self.poly_size = self.degree + 1
        self.curr_weight = configs['factor']
        self.prev_weight = 1.0 - self.curr_weight
        self.n_iter = configs["n_iter"]
        self.min_inliers = configs["min_inliers"]
        self.threshold = configs["threshold"]
        self.prev_points = {"left": None, "right": None}
        self.prev_coeffs = {"left": None, "right": None}
        self.judge = RegressionJudge()

    
    def fit(self, lanes):
        fit = []

        for i, lane in enumerate(lanes):
            # Direction variable
            direction = "left" if i == 0 else "right"
            
            if len(lane) < 2:
                print(f"WARNING: {direction} lane does not have enough points to perform fit. Skipping lane of length {len(lane)}.")
                continue

            # Generate inputs for polyfit
            X, y, threshold, params = self._get_inputs(lane)

            # Get best coeffs
            best_coeffs = self._get_best_fit(X, y, threshold)

            points = self._gen_line(best_coeffs, direction, params)
            
            fit.append(points)

        return fit
    
    def _get_inputs(self, lane):
        X = lane[:, 0]
        y = lane[:, 1]

        X, y, thresh, params = self._min_max_scaler(X, y)
        
        X_vars = [np.ones_like(X)]
        
        for i in range(1, self.degree + 1):
            X_vars.append(X**i)
        X = np.column_stack(X_vars)

        return X, y, thresh, params

    def _get_best_fit(self, X, y, threshhold):

        if self.method == "OLS":
            return self._calc_coeffs(X, y)
        
        # Create variables
        best_inliers = None
        best_inlier_count = 0
        best_coeffs = None
        n_points = len(X)
        sample_size = min(max(self.degree+3, 5), n_points)
        
        for _ in range(self.n_iter):

            # Random sampling
            sample_idx = np.random.choice(n_points, size=sample_size, replace=False)
            sample_X = X[sample_idx]
            sample_y = y[sample_idx]

            # Fit polynomial to samples
            try:
                coeffs = self._calc_coeffs(sample_X, sample_y)


                if not isinstance(coeffs, np.ndarray) or len(coeffs) != self.poly_size:
                    continue

            except (np.linalg.LinAlgError, TypeError, ValueError):
                continue

            # Evaluate sample fit on all points
            y_pred = self._poly_val(coeffs, X)
            errors = np.abs(sum(y - y_pred) ** 2)

            # Count inliers (points close to fit)
            inliers = errors < threshhold
            inlier_count = np.sum(inliers)

            # Best coeffs check
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_coeffs = coeffs

        # Leading Coefficient check: If leading coefficient is a negative value, fit all data
        if self.degree == 2 and best_coeffs is not None:
            if best_coeffs[0] < 0:
                print(f"WARNING: Suspecious parabola a = {best_coeffs[0]}")
                return self._calc_coeffs(X, y)

        # Best coeffs vs normal polyfit check
        if best_inliers is not None and best_inlier_count >= self.min_inliers * n_points:
            inlier_X = X[best_inliers]
            inlier_y = y[best_inliers]
            if len(inlier_X) >= self.poly_size:
                try:
                    ransac_coeffs = self._calc_coeffs(inlier_X, inlier_y)
                    if isinstance(ransac_coeffs, np.ndarray) and len(ransac_coeffs) == self.poly_size:
                        return ransac_coeffs
                except (np.linalg.LinAlgError, TypeError, ValueError):
                    pass
                else:
                    return best_coeffs # Return best coeffs without refit

        # FAIL SAFE: Fit all data
        try:
            last_resort = self._calc_coeffs(X, y)
            if isinstance(last_resort, np.ndarray) and len(last_resort) == self.poly_size:
                return last_resort
        except:
            pass

    def _gen_line(self, coeffs, direction, scale_params):

        prev_coeffs = self.prev_coeffs.get(direction, None)
        
        # Smooth coeffs, if able
        coeffs = self._exp_moving_avg(prev_coeffs, coeffs)

        # Assign coeffs to persistent variable for next frame
        self.prev_coeffs[direction] = coeffs

        # Generate 100 points in scaled space
        X_scaled = np.linspace(0, 1, 100)

        # Estimate respective y-values in scaled space
        y_scaled = self._poly_val(coeffs, X_scaled)

        # Inverse random X-values and predicted y-values back to normal space
        X, y = self._inverse_scaler(X_scaled, y_scaled, scale_params)

        # Convert X, y to a numpy array
        points = np.array([X, y], dtype=np.float32).T

        prev_points = self.prev_points.get(direction, None)

        # Smooth points, if able
        points = self._exp_moving_avg(prev_points, points)
    
        # Assign points to persistent variable for next frame
        self.prev_points[direction] = points

        return points.astype(np.float32)

    def _calc_coeffs(self, X, y):
        '''(X.T * X)**-1 * (X.T & y)'''
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        else:
            XT = X.T
            A = (XT @ X)
            b = XT @ y
            return np.linalg.solve(A, b)
        
    def _poly_val(self, coeffs, X):
        b0 = coeffs[0]
        return b0 + sum([X**(i+1) * b for i, b in enumerate(coeffs[1:])])

    def _exp_moving_avg(self, prev, curr):
        if prev is None:
            return curr
        return (self.curr_weight * curr + self.prev_weight * prev)
    
    def _min_max_scaler(self, X, y):
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        
        targets = [X, y, self.threshold]

        params = []
        for i in range(2):
            val = targets[i]
            min, max = val.min(), val.max()
            params.append(min), params.append(max)
            targets[i] = (val - min) / (max - min) if max > min else val
        return targets[0], targets[1], targets[2], params[:4]
    
    def _inverse_scaler(self, X_scaled, y_scaled, scale_params):
        X_min, X_max, y_min, y_max = scale_params
        X = X_scaled * (X_max - X_min) + X_min
        y = y_scaled * (y_max - y_min) + y_min
        return X, y