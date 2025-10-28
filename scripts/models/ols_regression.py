import numpy as np
from .evaluation import RegressionJudge

class OLSRegression:
    '''Test'''
    
    def __init__(self):

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

            X_scaled, y_scaled, params = self._gen_inputs(X, y)

            coeffs = self._calc_coeffs(X_scaled, y_scaled)

            line = self._gen_line(coeffs, direction, scale_params=params)
            
            self.judge.evaluate(y, line[:, 1], direction)

            fit.append(line)

        return np.array(fit) if len(fit) > 0 else None

    def _gen_line(self, coeffs, direction, scale_params):
        prev_coeffs = self.prev_coeffs.get(direction, None)

        if prev_coeffs is not None:
            coeffs = (0.6 * coeffs + (1 - 0.6) * prev_coeffs)

        self.prev_coeffs[direction] = coeffs

        X_scaled = np.linspace(0, 1, 100)
        y_scaled = self._poly_val(coeffs, X_scaled)

        X, y = self._inverse_scaler(X_scaled, y_scaled, scale_params)
        points = np.array([X, y], dtype=np.int32).T

        # Smooth points if able
        prev_points = self.prev_points.get(direction, None)

        if prev_points is not None:
            points = (0.6 * points + (1 - 0.6) * prev_points).astype(np.int32)
        
        # Assign points to persistent variable for next frame
        self.prev_points[direction] = points

        return points
        
    def _calc_coeffs(self, X, y):
        '''(X.T * X)**-1 * (X.T & y)'''
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        else:
            X_t = X.T
            X_t_X = np.dot(X_t, X)
            X_t_X_inv = np.linalg.inv(X_t_X)

            X_t_y = np.dot(X_t, y)            
            return np.dot(X_t_X_inv, X_t_y)
        
    def _poly_val(self, coeffs, X):
        if len(coeffs) == 2:
            b0, b1 = coeffs
            return b1 * X + b0
        elif len(coeffs) == 3:
            b0, b1, b2 = coeffs
            return b0 + b1 * X + b2 * X**2
        else:
            raise ValueError(f"ERROR: Coeffs ({coeffs}) length not 2 or 3.")

    def _gen_inputs(self, X, y):

        X, y, params = self._min_max_scaler(X, y)
        
        y_range = y.max() - y.min()

        top_third = y[y < (y.min() + y_range / 3)]

        if len(top_third) < 10:
            degree = 1
        else:
            degree = 2
        
        X_vars = [np.ones_like(X)]
        
        for i in range(1, degree + 1):
            X_vars.append(X**i)
        X = np.column_stack(X_vars)

        return X, y, params


    def _min_max_scaler(self, X, y):
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")

        targets = [X, y]
        params = []
        for i in range(len(targets)):
            val = targets[i]
            min, max = val.min(), val.max()
            params.append(min), params.append(max)
            targets[i] = (val - min) / (max - min) if max > min else val

        return targets[0], targets[1], params
    
    def _inverse_scaler(self, X_scaled, y_scaled, scale_params):
        X_min, X_max, y_min, y_max = scale_params

        X = X_scaled * (X_max - X_min) + X_min
        y = y_scaled * (y_max - y_min) + y_min
        return X, y
        
