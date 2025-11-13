import numpy as np
from numpy.typing import NDArray

class OLSRegression:
    '''Test'''
    
    def __init__(self, degree:int = 2):
        self.degree = degree
        self.poly_size = self.degree + 1
        self.inlier_ratio = 1.0
        self.name = "OLS Regression"

    def fit_predict(self, X, y):
        coeffs = self.fit(X, y)
        return self.predict(coeffs)
    
    def fit(self, X:NDArray, y:NDArray):
        # Generate X matrix
        X_mat = self._gen_X_design(X)

        self.fitted_X, self.fitted_y = X, y
        
        # Estimate best coeffs
        return self._calc_coeffs(X_mat, y)
    
    def predict(self, coeffs:NDArray):
        # Generate 100 points in scaled space
        X_lin = np.linspace(0, 1, 100)

        # Estimate respective y-values in scaled space
        y_pred = self.poly_val(coeffs, X_lin)

        return X_lin, y_pred
        
    def _calc_coeffs(self, X:NDArray, y:NDArray):
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
        
    def poly_val(self, coeffs:NDArray, X:NDArray):
        result = coeffs[-1]
        for coef in reversed(coeffs[:-1]):
            result = result * X + coef
        return result
    
    def _gen_X_design(self, X:NDArray):
        X_mat = [np.ones_like(X)]
        
        for i in range(1, self.poly_size):
            X_mat.append(X**i)
        X = np.column_stack(X_mat)

        return X

    def get_fitted_X_y(self):
        return self.fitted_X, self.fitted_y