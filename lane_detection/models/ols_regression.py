import numpy as np
from lane_detection.utils import MinMaxScaler

class OLSRegression:
    '''Test'''
    
    def __init__(self, fps:int=None, degree:int = 2):
        self.degree = degree
        self.poly_size = self.degree + 1
        self.inlier_ratio = 1.0
        self.max_error = None
        self.fps = fps
        self.name = "OLS Regression"
    
    def fit(self, X, y, y_range=None, direction:str=None):
        # Generate X matrix
        X_mat = self._gen_X_design(X)
        
        # Estimate best coeffs
        return self._calc_coeffs(X_mat, y)
    
    def predict(self, coeffs, n):
        # Generate 100 points in scaled space
        X_lin = np.linspace(0, 1, n)

        # Estimate respective y-values in scaled space
        y_pred = self._poly_val(coeffs, X_lin)

        return X_lin, y_pred
        
    def _calc_coeffs(self, X:np.ndarray, y:np.ndarray):
        '''(X.T * X)**-1 * (X.T & y)'''
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        else:
            XT = X.T
            A = XT @ X

            if np.linalg.cond(A) > 1e10:
                raise np.linalg.LinAlgError("Matrix is ill-conditioned")
            b = XT @ y
            return np.linalg.solve(A, b).astype(np.ndarray)
        
    def _poly_val(self, coeffs, X):
        result = coeffs[-1]
        for coef in reversed(coeffs[:-1]):
            result = result * X + coef
        return result
    
    def _gen_X_design(self, X):
        X_mat = [np.ones_like(X)]
        
        for i in range(1, self.poly_size):
            X_mat.append(X**i)
        X = np.column_stack(X_mat)

        return X
    
    def _update_fps(self, fps):
        self.fps = fps