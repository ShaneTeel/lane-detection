import numpy as np
from typing import Literal
from lane_detection.models.ransac_regression import RANSACRegression
from lane_detection.models.ols_regression import OLSRegression

class KalmanFilteredRANSAC():

    def __init__(self, degree:int=2, n_iter:int=50, min_inliers:float=0.8, max_error:int=10, environment:Literal["city", "highway"]="highway", fps:int=None):
        self.estimator = RANSACRegression(degree, n_iter, min_inliers, max_error)
        self.poly_size = self.estimator.poly_size
        self.kalman_left = None
        self.kalman_right = None
        self.fps = fps
        self.environment = environment
        self.name = "Kalman Filtered OLS Regression"

    def fit(self, X, y, y_range:float, direction:str):
        coeffs = self.estimator.fit(X, y, y_range)

        if direction == "left" and self.kalman_left is None:
            self.kalman_left = KalmanFilter(self.fps, coeffs, self.environment)
        elif direction == "right" and self.kalman_right is None:
            self.kalman_right = KalmanFilter(self.fps, coeffs, self.environment)
        
        kalman = self.kalman_left if direction == "left" else self.kalman_right

        kalman.predict()

        R = kalman._compute_R(self.estimator.inlier_ratio)

        kalman.update(coeffs, R)

        return kalman.get_coeffs()
    
    def predict(self, coeffs):
        return self.estimator.predict(coeffs)
    
    def _update_fps(self, fps):
        self.fps = fps
        
    def _get_fitted_X_y(self):
        return self.estimator._get_fitted_X_y()
    
    def _poly_val(self, coeffs, X):
        return self.estimator._poly_val(coeffs, X)
    
class KalmanFilteredOLS():

    def __init__(self, fps:int=None, degree:int=2, environment:Literal["city", "highway"]="highway"):
        self.estimator = OLSRegression(degree)
        self.poly_size = self.estimator.poly_size
        self.kalman_left = None
        self.kalman_right = None
        self.fps = fps
        self.environment = environment
        self.name = "Kalman Filtered OLS Regression"

    def fit(self, X, y, y_range:float, direction:str):
        coeffs = self.estimator.fit(X, y)

        if direction == "left" and self.kalman_left is None:
            self.kalman_left = KalmanFilter(self.fps, coeffs, self.environment)
        elif direction == "right" and self.kalman_right is None:
            self.kalman_right = KalmanFilter(self.fps, coeffs, self.environment)
        
        kalman = self.kalman_left if direction == "left" else self.kalman_right

        kalman.predict()

        R = kalman._compute_R(self.estimator.inlier_ratio)

        kalman.update(coeffs, R)

        return kalman.get_coeffs()
    
    def predict(self, coeffs):
        return self.estimator.predict(coeffs)
    
    def _update_fps(self, fps):
        self.fps = fps

    def _get_fitted_X_y(self):
        return self.estimator._get_fitted_X_y()
    
    def _poly_val(self, coeffs, X):
        return self.estimator._poly_val(coeffs, X)
    
class KalmanFilter():

    def __init__(self, fps, coeffs, environment:Literal["city", "highway"]):
        self.dt = 1 / fps
        self.x = self._initialize_current_state(coeffs)
        self.P = np.eye(len(self.x)) * 10. # Convert `10.` to input argument / class param
        self.I = np.eye(len(self.x))
        self.Q = self._initialize_model_uncertainty_matrix(environment)
        self.F, self.H = self._initialize_F_H_matrices()
        self.F_T = self.F.T
        self.H_T = self.H.T

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F_T + self.Q

    def update(self, coeffs:np.ndarray, R:np.matrix):
        z = coeffs.reshape(-1, 1)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H_T + R
        try:
            K = self.P @ self.H_T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("WARNING: `S` is singular, using pseudo-inverse")
            K = self.P @ self.H_T @ np.linalg.pinv(S)

        self.x = self.x + K @ innovation
        IKH = self.I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def get_coeffs(self):
        return self.x[:len(self.x)//2].flatten()
    
    def _initialize_current_state(self, coeffs:np.ndarray):
        top = coeffs.reshape(-1, 1)
        bottom = np.zeros_like(top)
        return np.block([
            [top],
            [bottom]
        ]).astype(float)
    
    def _initialize_model_uncertainty_matrix(self, environment):
        if environment == "highway":
            return np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        return np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    def _initialize_F_H_matrices(self):
        I = np.eye(len(self.x) // 2, dtype=float)
        dt_I = self.dt * I
        zeros = np.zeros_like(I, dtype=float)
        F = np.block([
            [I, dt_I],
            [zeros, I]
        ])
        H = np.block([
            [I, zeros]
        ])
        return F, H

    def _compute_R(self, inlier_ratio):
        # Change this to dynamically up
        if inlier_ratio > 0.8:
            return np.diag([2.0, 2.0, 1.0])
        elif inlier_ratio > 0.5:
            return np.diag([5.0, 5.0, 2.0])
        else:
            return np.diag([20.0, 20.0, 10.0])