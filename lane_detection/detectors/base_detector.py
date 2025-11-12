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
        evaluation_report_name = f"{self.studio.source.name} {self.estimator.name}"
        if view_style is not None:
            win_name = f"{evaluation_report_name} {view_style.capitalize()} View"
            cv2.namedWindow(win_name)

        if save or view_style is not None:
            frame_names = self.studio._get_frame_names(view_style)

        if self.studio.source.source_type != "image" and view_style is not None:
            self.studio.playback.print_playback_menu()
        
        if save:
            self.studio.write._initialize_writer()
        
        while True:
            ret, frame = self.studio.return_frame()
            if not ret:
                break
            else:
                masked = self.mask.inverse_mask(frame)
                thresh, edge_map, kps = self.preprocessor.transform(masked, self.mask.x_mid)
                lane_lines = []
                for i, lane in enumerate(kps):

                    # Direction variable
                    direction = "left" if i == 0 else "right"
                    if len(lane) < self.estimator.poly_size:
                        print(f"WARNING: {direction} lane does not have enough points to perform fit. Skipping lane of length {len(lane)}.")
                        continue

                    # Generate inputs
                    X = lane[:, 0]
                    y = lane[:, 1]

                    # Scale X, y
                    X_scaled, y_scaled = self.scaler.transform(X, y)
                    n = len(y)
                    y_range = self.scaler.y_max - self.scaler.y_min

                    # Estimate coeffs, generate X, predict y
                    coeffs = self.estimator.fit(X_scaled, y_scaled, y_range, direction)
                    X_lin_scaled, y_pred_scaled = self.estimator.predict(coeffs, n)
                    
                    # Inverse scale, Evaluate fit, and create points
                    X_lin, y_pred = self.scaler.inverse_transform(X_lin_scaled, y_pred_scaled)
                    self.evaluate.evaluate(y, y_pred, direction)

                    points = np.array([X_lin, y_pred], dtype=np.int32).T

                    lane_lines.append(points)

                if save or view_style is not None:
                    frame_lst = [frame, thresh, edge_map]
                    final = self.studio.gen_view(frame_lst, frame_names, lane_lines, view_style, stroke=stroke, fill=fill)

                if view_style is not None:
                    cv2.imshow(win_name, final)
    
                if save:
                    self.studio.write.writer.write(final)

                if self.studio.playback.playback_controls():
                    break

        self.evaluate.regression_report(evaluation_report_name)