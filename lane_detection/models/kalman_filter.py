import numpy as np
from numpy.typing import NDArray
    
class KalmanFilter():

    def __init__(self, fps:int, coeffs:NDArray):
        self.dt = 1 / fps
        self.x = self._initialize_current_state(coeffs)
        self.P = np.eye(len(self.x)) * 10. # Convert `10.` to input argument / class param
        self.I = np.eye(len(self.x))
        self.Q = self._initialize_Q_matrix()
        self.F, self.H = self._initialize_F_H_matrices()
        self.F_T = self.F.T
        self.H_T = self.H.T

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F_T + self.Q

    def update(self, coeffs:NDArray, R:np.matrix):
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
    
    def _initialize_current_state(self, coeffs:NDArray):
        top = coeffs.reshape(-1, 1)
        bottom = np.zeros_like(top)
        return np.block([
            [top],
            [bottom]
        ]).astype(float)
    
    def _initialize_Q_matrix(self):
        return np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])

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

    def compute_R(self, inlier_ratio:float):
        # Change this to dynamically up
        if inlier_ratio > 0.8:
            return np.diag([2.0, 2.0, 1.0])
        elif inlier_ratio > 0.5:
            return np.diag([5.0, 5.0, 2.0])
        else:
            return np.diag([20.0, 20.0, 10.0])