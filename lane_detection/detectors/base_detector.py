import cv2
import numpy as np
from typing import Literal
from lane_detection.studio import StudioManager
from lane_detection.utils import Evaluator, ROISelector

class BaseDetector():

    def __init__(self, source, preprocessor, estimator, roi:np.ndarray, stroke_color:tuple=(0, 0, 255), fill_color:tuple=(0, 255, 0)):        
        self.studio = StudioManager(source, stroke_color, fill_color)
        self.mask = ROISelector(roi)
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.metrics = Evaluator()

    def detect(self, view_style: Literal[None, "inset", "mosaic", "composite"]="inset", stroke:bool=False, fill:bool=True, save:bool=False):        
        win_name = f"{self.studio.source.name} {view_style.capitalize()} View"
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
                thresh, edge_map, kps = self.preprocessor.preprocess(masked, self.mask.x_mid)
                lane_lines = []
                for i, lane in enumerate(kps):

                    # Direction variable
                    direction = "left" if i == 0 else "right"
                    if len(lane) < self.estimator.ols.poly_size:
                        print(f"WARNING: {direction} lane does not have enough points to perform fit. Skipping lane of length {len(lane)}.")
                        continue

                    # Generate inputs
                    X = lane[:, 0]
                    y = lane[:, 1]

                    X_scaled, y_scaled, max_error_scaled, params = self._min_max_scaler(X, y, self.estimator.max_error)

                    # Estimate coeffs, generate X, predict y
                    coeffs = self.estimator.fit(X_scaled, y_scaled, max_error_scaled)
                    X_lin, y_pred = self.estimator.predict(coeffs)
                    
                    # Inverse scale, and create points
                    X, y = self._inverse_scaler(X_lin, y_pred, params)
                    points = np.array([X, y], dtype=np.int32).T

                    lane_lines.append(points)

                frame_lst = [frame, thresh, edge_map]
                final = self.studio.gen_view(frame_lst, frame_names, lane_lines, view_style, stroke=stroke, fill=fill)

                cv2.imshow(win_name, final)

                if save:
                    self.studio.write.writer.write(final)

                if self.studio.playback.playback_controls():
                    break

    def _min_max_scaler(self, X, y, max_error:float=None):
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        
        targets = [X, y, max_error]

        params = []
        for i in range(2):
            val = targets[i]
            v_min, v_max = val.min(), val.max()
            params.append(v_min), params.append(v_max)
            targets[i] = (val - v_min) / (v_max - v_min) if v_max > v_min else val
            if i == 1 and max_error is not None:
                val = targets[2]
                targets[2] = val / (v_max - v_min) if v_max > v_min else val
        return targets[0], targets[1], targets[2], params[:4]
    
    def _inverse_scaler(self, X_scaled, y_scaled, scale_params):
        X_min, X_max, y_min, y_max = scale_params
        X = X_scaled * (X_max - X_min) + X_min
        y = y_scaled * (y_max - y_min) + y_min
        return X, y

if __name__=="__main__":
    from lane_detection.models import RANSACRegression
    from lane_detection.preprocessing import HoughFeatureEngineer

    src = "../../../media/in/lane1-straight.mp4"
    # src = "../../../media/in/test_img1.jpg"

    roi = np.array([[[75, 540], 
                     [925, 540], 
                     [520, 320], 
                     [450, 320]]], dtype=np.int32)
    
    estimator = RANSACRegression()
    preprocessor = HoughFeatureEngineer()

    hough = LaneDetector(src, preprocessor, estimator, roi)

    hough.detect("mosaic", stroke=True, fill=True)