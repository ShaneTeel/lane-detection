import numpy as np

class OLSRegression:
    '''Test'''
    
    def __init__(self, degree:int = 2):
        self.degree = degree
        self.poly_size = self.degree + 1
        self.sample_size = self.poly_size + 2

    def fit(self, X:np.ndarray, y:np.ndarray):
        return self._calc_coeffs(X, y)
    
    def predict(self, coeffs):
        # Generate 100 points in scaled space
        X_lin = np.linspace(0, 1, 100)

        # Estimate respective y-values in scaled space
        y_pred = self._poly_val(coeffs, X_lin)

        return X_lin, y_pred
        
    def _calc_coeffs(self, X, y):
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