import cv2
import numpy as np
from typing import Literal
from lane_detection.studio import StudioManager
from lane_detection.utils import RegressionEvaluator, ROISelector, MinMaxScaler

class BaseDetector():

    def __init__(self, source, preprocessor, estimator, roi:np.ndarray, stroke_color:tuple=(0, 0, 255), fill_color:tuple=(0, 255, 0)):        
        self.studio = StudioManager(source, stroke_color, fill_color)
        self.mask = ROISelector(roi)
        self.scaler = MinMaxScaler()
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.estimator._update_fps(self.studio.source.fps)
        self.evaluate = RegressionEvaluator()

    def detect(self, view_style: Literal[None, "inset", "mosaic", "composite"]="inset", stroke:bool=False, fill:bool=True, save:bool=False):
        report_name = f"{self.studio.source.name} {self.estimator.name}"
        frame_names = self._configure_output(report_name, view_style, save)

        while True:
            ret, frame = self.studio.return_frame()
            if not ret:
                break
            else:
                thresh, edge_map, kps = self.preprocess(frame)
                lane_lines = []
                for i, lane in enumerate(kps):

                    # Direction variable
                    direction = "left" if i == 0 else "right"
                    if len(lane) < self.estimator.poly_size:
                        print(f"WARNING: {direction} lane does not have enough points to perform fit. Skipping lane of length {len(lane)}.")
                        continue

                    X_scaled, y_scaled, y_range = self.generate_inputs(lane)
                    coeffs, X_100, y_pred_100 = self.fit_predict(X_scaled, y_scaled, y_range, direction)

                    # Format and append points
                    points = self.generate_final_points(coeffs, X_100, y_pred_100)
                    lane_lines.append(points)

                    # Evaluate fit, predict
                    self.evaluate_fit_predict(coeffs, direction)

                frame_lst = [frame, thresh, edge_map]
                final = self.studio.gen_view(frame_lst, frame_names, lane_lines, view_style, stroke, fill)
                
                if view_style is not None:
                    cv2.imshow(self.win_name, final)
                    if self.studio.playback.playback_controls():
                        break
                if save:
                    self.studio.write.write(final)

        self.evaluate.regression_report(report_name)

    def fit_predict(self, X, y, y_range:float=None, direction:str=None):
        # Estimate coeffs, Generate X, predict y
        coeffs = self.estimator.fit(X, y, y_range, direction)
        X_100, y_pred_100 = self.estimator.predict(coeffs)
        return coeffs, X_100, y_pred_100
    
    def generate_inputs(self, lane):
        # Generate inputs
        X = lane[:, 0]
        y = lane[:, 1]

        # Scale X, y; calc y_range
        X_scaled, y_scaled = self.scaler.transform(X, y)
        y_range = self.scaler.y_max - self.scaler.y_min
        return X_scaled, y_scaled, y_range
    
    def generate_final_points(self, coeffs, X, y):
        X_final, y_final = self.scaler.inverse_transform(X, y)
        
        return np.array([X_final, y_final], dtype=np.int32).T
    
    def evaluate_fit_predict(self, coeffs, direction):
        fitted_X, fitted_y = self.estimator._get_fitted_X_y()
        y_pred_scaled = self.estimator._poly_val(coeffs, fitted_X)
        X_original, y_original = self.scaler.inverse_transform(fitted_X, fitted_y)
        _, y_pred = self.scaler.inverse_transform(X_original, y_pred_scaled)
        self.evaluate.evaluate(y_original, y_pred, direction)

    def preprocess(self, frame):
        masked = self.mask.inverse_mask(frame)
        return self.preprocessor.transform(masked, self.mask.x_mid)
    
    def _configure_output(self, report_name:str, view_style:str=None, save:bool=False):
        if view_style is not None:      
            self.win_name = f"{report_name} {view_style.capitalize()} View"
            cv2.namedWindow(self.win_name)

            if self.studio.source.source_type != "image":
                self.studio.playback.print_playback_menu()
            if save:
                self.studio.write._initialize_writer()
            return self.studio._get_frame_names(view_style.lower())
        