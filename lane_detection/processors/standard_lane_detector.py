import numpy as np
from numpy.typing import NDArray
from lane_detection.utils.scaler import MinMaxScaler
from lane_detection.utils.evaluator import RegressionMetrics
from lane_detection.models.kalman_filter import KalmanFilter
from lane_detection.models.ransac_regression import OLSRegression

class BaseLaneDetector():

    def __init__(self, degree:int=2, n_iter:int=50, min_inliers:float=0.8, max_error:int=10, fps:int=None):
        self.kalman = None
        self.fps = fps
        self.degree = degree
        self.n_iter = n_iter
        self.min_inliers = min_inliers
        self.max_error = max_error
        self.scaler = None
        self.name = "Kalman Filtered RANSAC Regression"
        self.evaluator = RegressionMetrics(self.name)

    def process_lane(self, lane:NDArray):
        if len(lane) < self.degree + 1:
            print(f"WARNING: lane does not have enough points to perform fit. Skipping lane of length {len(lane)}.")
            return np.array([])

        # Scale and fit points
        X, y = self.fit_transform(lane)
        coeffs = self.fit(X, y)

        # Generate clean X and predict y; inverse transform to normal space
        X_lin, y_pred = self.predict(coeffs)
        X_fin, y_fin = self.inverse_transform(X_lin, y_pred)
        
        # Evaluate fit
        fitted_X, fitted_y = self.get_fitted_X_y()
        self.evaluate(coeffs, fitted_X, fitted_y)

        return np.array([X_fin, y_fin], dtype=np.int32).T

    def fit_transform(self, lane:NDArray):

        X = lane[:, 0]
        y = lane[:, 1]

        self.scaler = MinMaxScaler()

        X, y = self.scaler.fit_transform(X, y)
        return X, y

    def fit(self, X:NDArray, y:NDArray):

        y_range = self.scaler.y_max - self.scaler.y_min
        max_error = self.max_error / y_range if y_range != 0 else self.max_error

        self.estimator = OLSRegression(self.degree, self.n_iter, self.min_inliers, max_error)
        coeffs = self.estimator.fit(X, y)

        if self.kalman is None:
            self.kalman = KalmanFilter(self.fps, coeffs)

        self.kalman.predict()

        R = self.kalman.compute_R(self.estimator.inlier_ratio)

        self.kalman.update(coeffs, R)

        return self.kalman.get_coeffs()
    
    def predict(self, coeffs:NDArray):
        return self.estimator.predict(coeffs)
    
    def inverse_transform(self, X:NDArray, y:NDArray):
        return self.scaler.inverse_transform(X, y)
    
    def get_fitted_X_y(self):
        return self.estimator.get_fitted_X_y()
    
    def evaluate(self, coeffs:NDArray, X:NDArray, y:NDArray):
        y_pred_scaled = self.estimator.poly_val(coeffs, X)
        y_true = self.scaler._inverse_transform_y(y)
        y_pred = self.scaler._inverse_transform_y(y_pred_scaled)
        self.evaluator.compute_metrics(y_true, y_pred)
    
    def return_evaluator(self):
        return self.evaluator
