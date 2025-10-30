import numpy as np
from .metrics import RegressionJudge

class OLSRegression:
    '''Test'''
    _DEFAULT_CONFIGS = {
        "estimator": {"degree": 2, "factor":0.6, "n_iter": None, "min_inliers": None, "threshold": None}
    }
    
    def __init__(self, configs:dict=None):
        if configs is None:
            configs = self._DEFAULT_CONFIGS['estimator']

        self.degree = configs['degree']
        self.curr_weight = configs['factor']
        self.prev_weight = 1 - self.curr_weight
        self.n_iter = configs['n_iter']
        self.min_inliers = configs['min_inliers']
        self.threshold = configs['threshold']
        self.prev_points = {"left": None, "right": None}
        self.prev_coeffs = {"left": None, "right": None}
        self.judge = RegressionJudge()
    
    def fit(self, lanes):
        if lanes is None:
            print("WARNING: Argument passed to lines is None. Skipping line")
            return None
        
        fit = []
        for i, lane in enumerate(lanes):
            direction = "left" if i == 0 else "right"

            if lane is None:
                print(f"WARNING: {direction} lane is None. Skipping")
                continue
            X = lane[:, 0]
            y = lane[:, 1]
            n_points = len(y)

            X_scaled, y_scaled, params = self._gen_inputs(X, y)

            coeffs = self._calc_coeffs(X_scaled, y_scaled)

            line = self._gen_line(coeffs, direction, params)

            y_pred = self._gen_y_pred(direction, n_points, params)
            
            self.judge.evaluate(y, y_pred, direction)

            fit.append(line)

        return np.array(fit, dtype=np.float32)
    
    def _gen_inputs(self, X, y):

        X, y, params = self._min_max_scaler(X, y)
        
        X_vars = [np.ones_like(X)]
        
        for i in range(1, self.degree + 1):
            X_vars.append(X**i)
        X = np.column_stack(X_vars)

        return X, y, params
    
    def _calc_coeffs(self, X, y):
        '''(X.T * X)**-1 * (X.T & y)'''
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        else:
            XT = X.T
            A = (XT @ X)
            b = XT @ y
            return np.linalg.solve(A, b)
    
    def _gen_line(self, coeffs, direction, scale_params):
        prev_coeffs = self.prev_coeffs.get(direction, None)

        coeffs = self._exp_moving_avg(prev_coeffs, coeffs)

        self.prev_coeffs[direction] = coeffs

        X_scaled = np.linspace(0, 1, 100)
        y_scaled = self._poly_val(coeffs, X_scaled)

        X, y = self._inverse_scaler(X_scaled, y_scaled, scale_params)

        points = np.array([X, y], dtype=np.float32).T

        # Smooth points if able
        prev_points = self.prev_points.get(direction, None)

        points = self._exp_moving_avg(prev_points, points)
    
        # Assign points to persistent variable for next frame
        self.prev_points[direction] = points

        return points.astype(np.float32)

    def _gen_y_pred(self, direction, n_points, scale_params):
        coeffs = self.prev_coeffs.get(direction, None)

        X_scaled = np.linspace(0, 1, n_points)
        y_scaled = self._poly_val(coeffs, X_scaled)

        return self._inverse_scaler(X_scaled, y_scaled, scale_params)[1]
        
    def _poly_val(self, coeffs, X):
        b0 = coeffs[0]
        return b0 + sum([X**(i+1) * b for i, b in enumerate(coeffs[1:])])

    def _exp_moving_avg(self, prev, curr):
        if prev is None:
            return curr
        return (self.curr_weight * curr + self.prev_weight * prev)

    def _min_max_scaler(self, X, y, thresh:int=None):
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        
        targets = [X, y]

        if thresh is not None:
            targets.append(thresh)

        params = []
        for i in range(2):
            val = targets[i]
            min, max = val.min(), val.max()
            params.append(min), params.append(max)
            targets[i] = (val - min) / (max - min) if max > min else val
        return targets[0], targets[1], params[:4]
    
    def _inverse_scaler(self, X_scaled, y_scaled, scale_params):
        X_min, X_max, y_min, y_max = scale_params
        X = X_scaled * (X_max - X_min) + X_min
        y = y_scaled * (y_max - y_min) + y_min
        return X, y