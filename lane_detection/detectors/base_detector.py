import cv2
import numpy as np
from typing import Literal
from lane_detection.studio import StudioManager
from lane_detection.utils import Evaluator, ROISelector, MinMaxScaler

class BaseDetector():

    def __init__(self, source, preprocessor, estimator, roi:np.ndarray, stroke_color:tuple=(0, 0, 255), fill_color:tuple=(0, 255, 0)):        
        self.studio = StudioManager(source, stroke_color, fill_color)
        self.mask = ROISelector(roi)
        self.scaler = MinMaxScaler()
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.ransac = True if hasattr(self.estimator, "n_iter") else False
        self.metrics = Evaluator()

    def detect(self, view_style: Literal[None, "inset", "mosaic", "composite"]="inset", stroke:bool=False, fill:bool=True, save:bool=False):        
        win_name = f"{self.studio.source.name} {self.estimator.name} {view_style.capitalize()} View"
        cv2.namedWindow(win_name)
        frame_names = self.studio._get_frame_names(view_style)

        if self.studio.source.source_type != "image" or view_style is not None:
            self.studio.playback.print_playback_menu()
        
        if save:
            self.studio.write._initialize_writer()
        
        while True:
            ret, frame = self.studio.return_frame()
            if not ret:
                self.studio.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
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
                    X, y = self.scaler.transform(X, y)

                    # Estimate coeffs, generate X, predict y
                    if self.ransac:
                        y_range = self.scaler.y_max - self.scaler.y_min
                        max_error = np.abs(self.estimator.max_error / y_range)
                        coeffs = self.estimator.fit(X, y, max_error)
                    else:
                        coeffs = self.estimator.fit(X, y)

                    X_lin, y_pred = self.estimator.predict(coeffs)
                    
                    # Inverse scale, and create points
                    X, y = self.scaler.inverse_transform(X_lin, y_pred)
                    points = np.array([X, y], dtype=np.int32).T

                    lane_lines.append(points)

                frame_lst = [frame, thresh, edge_map]
                final = self.studio.gen_view(frame_lst, frame_names, lane_lines, view_style, stroke=stroke, fill=fill)

                cv2.imshow(win_name, final)

                if save:
                    self.studio.write.writer.write(final)

                if self.studio.playback.playback_controls():
                    break