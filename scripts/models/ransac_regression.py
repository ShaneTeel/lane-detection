import numpy as np
from .metrics import RegressionJudge

class RANSACRegression():
    '''Test'''

    _DEFAULT_CONFIG = {
        "estimator": {"degree": 2, "factor":0.6, "min_inliers": 0.3, "max_error": 20}
    }

    def __init__(self, configs:dict = None):
        if configs is None:
            configs = self._DEFAULT_CONFIG["estimator"]

        self.degree = configs['degree']
        self.poly_size = self.degree + 1
        self.curr_weight = configs['factor']
        self.prev_weight = 1.0 - self.curr_weight
        self.n_iter = configs["n_iter"]
        self.min_inliers = configs["min_inliers"]
        self.max_error = configs["max_error"]
        self.prev_points = {"left": None, "right": None}
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
            X, y, max_error, params = self._get_inputs(lane)

            # Get best coeffs
            best_coeffs = self._get_best_fit(X, y, max_error)

            points = self._gen_line(best_coeffs, direction, params)
            
            fit.append(points)

        return fit
    
    def _get_inputs(self, lane):
        X = lane[:, 0]
        y = lane[:, 1]

        X, y, max_error, params = self._min_max_scaler(X, y)
        
        X_mat = [np.ones_like(X)]
        
        for i in range(1, self.degree + 1):
            X_mat.append(X**i)
        X = np.column_stack(X_mat)

        return X, y, max_error, params

    def _get_best_fit(self, X, y, max_error):        
        # Create variables
        best_inliers = None
        best_inlier_count = 0
        best_coeffs = None
        population = len(X)

        # Determine consensus count. Support fraction (0<min_inliers<1) or absolute count (>=1).        
        if self.min_inliers is None:
            consensus = int(np.ceil(population * 0.5))
        else:
            try:
                if self.min_inliers < 1:
                    consensus = int(np.ceil(population * self.min_inliers))
                else:
                    consensus = int(self.min_inliers)
            except TypeError:
                # fallback to 50% if misconfigured
                consensus = int(np.ceil(population * 0.5))

        # If there are too few points to estimate the polynomial, fall back to OLS
        if population < self.poly_size:
            return self._calc_coeffs(X, y)

        # Use the minimal sample size necessary for the model (poly_size)
        sample_size = min(max(self.poly_size, 1), population)
        
        for _ in range(self.n_iter):
            # Random sampling
            sample_idx = np.random.choice(population, size=sample_size, replace=False)
            sample_X = X[sample_idx]
            sample_y = y[sample_idx]



            # Fit polynomial to samples
            try:
                coeffs = self._calc_coeffs(sample_X, sample_y)
                if not isinstance(coeffs, np.ndarray) or len(coeffs) != self.poly_size:
                    continue

            except (np.linalg.LinAlgError, TypeError, ValueError) as e:
                print(f"WARNING: polyfit error - {e}")
                continue

            # Evaluate sample fit on all points (original scaled X in column 1)
            y_pred = self._poly_val(coeffs, X[:, 1])

            # Use absolute error
            sample_errors = np.abs(y - y_pred)

            # Count inliers (points close to fit)
            # Ensure max_error is a scalar; if None, treat as very small to avoid accepting everything
            if max_error is None:
                threshold = 1e-6
            else:
                threshold = np.abs(max_error)

            inlier_mask = sample_errors <= threshold
            inlier_count = np.sum(inlier_mask)

            # Best coeffs check
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inlier_mask
                best_coeffs = coeffs.copy()
            
            if inlier_count == population:
                break
        try:
            frac = (best_inlier_count / population) * 100 if population > 0 else 0.0
        except Exception:
            frac = 0.0

        if best_inliers is not None and best_inlier_count >= consensus:
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
        if best_inliers is None or best_inlier_count < consensus:
            print(f"NO CONSENSUS! Best inlier's account for {frac}% of total population, but args required {self.min_inliers * 100}%. Falling back to full-data OLS.")
        # Leading Coefficient check: If leading coefficient is a negative value, fit all data
        if self.degree == 2 and best_coeffs is not None:
            if best_coeffs[-1] < 0:
                print(f"WARNING: Suspecious parabola a = {best_coeffs[-1]}")
                return self._calc_coeffs(X, y)

        # FAIL SAFE: Fit all data (ordinary least squares)
        try:
            last_resort = self._calc_coeffs(X, y)
            if isinstance(last_resort, np.ndarray) and len(last_resort) == self.poly_size:
                return last_resort
        except:
            pass

    def _gen_line(self, coeffs, direction, scale_params):
        # Generate 100 points in scaled space
        X_scaled = np.linspace(0, 1, 100)

        # Estimate respective y-values in scaled space
        y_scaled = self._poly_val(coeffs, X_scaled)

        # Inverse random X-values and predicted y-values back to normal space
        X, y = self._inverse_scaler(X_scaled, y_scaled, scale_params)

        # Convert X, y to a numpy array
        points = np.array([X, y], dtype=np.float32).T

        # Smooth points, if able
        prev_points = self.prev_points.get(direction, None)
        points = self._weighted_avg(prev_points, points)
        self.prev_points[direction] = points # assign points to instance to persist across runs

        return points.astype(np.int32)

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
        result = coeffs[-1]
        for coef in reversed(coeffs[:-1]):
            result = result * X + coef
        return result

    def _weighted_avg(self, prev, curr):
        if prev is None:
            return curr
        if curr is None:
            return prev
        
        return (self.curr_weight * curr + self.prev_weight * prev)
    
    def _min_max_scaler(self, X, y):
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        
        targets = [X, y, self.max_error]

        params = []
        for i in range(2):
            val = targets[i]
            v_min, v_max = val.min(), val.max()
            params.append(v_min), params.append(v_max)
            targets[i] = (val - v_min) / (v_max - v_min) if v_max > v_min else val
            if i == 1:
                val = targets[2]
                targets[2] = val / (v_max - v_min) if v_max > v_min else val
        return targets[0], targets[1], targets[2], params[:4]
    
    def _inverse_scaler(self, X_scaled, y_scaled, scale_params):
        X_min, X_max, y_min, y_max = scale_params
        X = X_scaled * (X_max - X_min) + X_min
        y = y_scaled * (y_max - y_min) + y_min
        return X, y