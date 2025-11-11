import numpy as np
from typing import Literal
np.set_printoptions(suppress=True)

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

    def _compute_R(self, inlier_ratio):
        # Change this to dynamically up
        if inlier_ratio > 0.8:
            return np.diag([2.0, 2.0, 1.0])
        elif inlier_ratio > 0.5:
            return np.diag([5.0, 5.0, 2.0])
        else:
            return np.diag([20.0, 20.0, 10.0])


if __name__=="__main__":
        
    coeffs = np.array([3, 4, 6]).reshape(-1, 1)

    test = KalmanFilter(30, coeffs, "highway")
    # Variable set-up
    # Current state
    x = np.array([[150.0], [2.3], [-0.05], [0.0], [0.0], [0.0]]) # Coeffs + velocities

    # Covariance Matrix (Uncertainty)
    P = np.eye(6) * 10.0 # Multiply by 10 to add uncertainty

    # Distance (FPS)
    dt = 1/30 # Frames per Second (FPS)
    I3 = np.eye(3, dtype=float)
    dt_I3 = dt * I3
    zeros = np.zeros_like(I3, dtype=float)
    F = np.block([
        [I3, dt_I3],
        [zeros, I3]
    ])

    # Process noise (model uncertainty)
    Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])

    # Measurement matrix
    H = np.block([
        [I3, zeros]
    ])

    # Process Frames
    # Step 1: Current State Prediction
    x_pred = F @ x
    print("\nStep 1: Current State Prediction (x_pred)")
    print(x_pred)

    # Step 2: Covariance Prediction
    P_pred = F @ P @ F.T + Q
    print("\nStep 2: Covariance Prediction (P_pred)")
    print(P_pred)

    # Step 3: Get Measurements (coeffs)
    z = np.array([[152.0], [2.4], [-0.05]])
    print("\nStep 3: Get measurements (coeffs)")
    print(z)

    # Get measurements noise
    R = np.diag([5.0, 5.0, 2.0])

    # Step 4: Innovation
    innovation = z - H @ x_pred
    print("\nStep 4: Innovation")
    print(innovation)

    # Step 5: Innovation Covariance
    S = H @ P_pred @ H.T + R
    print("\nStep 5: Innovation Covariance (S)")
    print(S)

    # Step 6: Kalman Gain
    K = P_pred @ H.T @ np.linalg.inv(S)
    print("\nStep 6: Kamlan Gain (K)")
    print(K)

    # Step 7: Current State Update
    x_updated = x_pred + K @ innovation
    print("\nStep 7: Updated State (x)")
    print(x_updated)

    # Step 8: Covariance Update
    I = np.eye(6)
    P_updated = (I - K @ H) @ P_pred
    print("\nStep 8: Updated P Diagnol")
    print(np.diag(P_updated))