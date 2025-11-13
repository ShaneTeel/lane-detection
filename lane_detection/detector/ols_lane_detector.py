import numpy as np
from numpy.typing import NDArray

from lane_detection.utils.scaler import MinMaxScaler
from lane_detection.utils.evaluator import RegressionMetrics
from lane_detection.models.ransac_regression import OLSRegression

class OLSLaneDetector():

    def __init__(self, degree:int=2):
        self.kalman = None
        self.degree = degree
        self.scaler = None
        self.name = f"OLS Regression"
        self.evaluator = RegressionMetrics(self.name)

    def detect(self, lane:NDArray):
        if len(lane) < self.degree + 1:
            print("WARNING: lane does not have enough points to perform fit")
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
        self.estimator = OLSRegression(self.degree)
        coeffs = self.estimator.fit(X, y)
        return coeffs
    
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

    def return_metrics(self):
        return self.evaluator.return_metrics()