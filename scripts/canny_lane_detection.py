import cv2
from typing import Literal
from numpy.typing import NDArray
from studio import StudioManager
from preprocessing import ConfigManager, ROISelector, FeatureEngineer, BEVTransformer
from models import RANSACRegression

class CannyLaneDetector():

    _VALID_SETUP = {
        "preprocessor": {
            "generator": {
                'in_range': {'lower_bounds': [0, 255], 'upper_bounds': [0, 255]},
                'canny': {'weak_edge': [0, 301], 'sure_edge': [0, 301], 'blur_ksize': [3, 15], "blur_order": ["before", "after"]}
            },
            "extractor": {"filter_type": ["median", "mean"], "n_std": [0.1, 5.0], "weight": [0, 100]}
        },
        "estimator": {"method": ["RANSAC", "OLS"], "degree": [1, 5], "factor":[0.0, 1.0], 'min_inliers': [0.0, 1.0], "max_error": [0, 100]}
    }

    _DEFAULT_CONFIGS = {
        "preprocessor": {
            "generator": {
                'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
                'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "after"}
            },
            'extractor': {"filter_type": "median", "n_std": 2.0, "weight": 5}
        },
        "estimator": {"method": "RANSAC", "degree": 2, "factor":0.6, "n_iter": 100, "min_inliers": 0.3, "max_error": 10}
    }

    def __init__(self, source, roi:NDArray, configs:dict=None, stroke_color:tuple=(0, 0, 255), fill_color:tuple=(0, 255, 0)):
        if configs is None:
            pre_configs, est_configs = self._DEFAULT_CONFIGS['preprocessor'], self._DEFAULT_CONFIGS["estimator"]
        else:
            pre_configs, est_configs = self._get_configs(configs, self._DEFAULT_CONFIGS, self._VALID_SETUP)
        
        self.studio = StudioManager(source, stroke_color, fill_color)
        self.mask = ROISelector(roi)
        self.bev = BEVTransformer(self.mask.roi, (self.studio.source.height, self.studio.source.width), self.mask.x_max, self.mask.y_max)
        self.preprocessor = FeatureEngineer(self.mask.x_mid, pre_configs)
        self.estimator = RANSACRegression(est_configs)
        
    def detect(self, view_style: Literal[None, "inset", "mosaic", "composite"]="inset", stroke:bool=False, fill:bool=True, save:bool=False):        
        win_name = f"{self.studio.source.name} {view_style} View"
        cv2.namedWindow(win_name)
        
        frame_names = self._get_frame_names(view_style)

        if self.studio.source.source_type != "image" or view_style is not None:
            self.studio.playback.print_playback_menu()
        
        if save:
            self.studio.write._initialize_writer()
        
        while True:
            ret, frame = self.studio.return_frame()
            if not ret:
                self.studio.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                masked = self.mask.inverse_mask(frame)
                warped = self.bev.transform(masked)
                thresh, edge_map, kps = self.preprocessor.generate_features(warped)
                warped_fit = np.array(self.estimator.fit(kps))
                normal_fit = self.bev.inverse_transform(warped_fit)
                frame_lst = [frame, thresh, edge_map]
                final = self.studio.gen_view(frame_lst, frame_names, normal_fit, view_style, stroke=stroke, fill=fill)

                cv2.imshow(win_name, final)

                if save:
                    self.studio.write.writer.write(final)

                if self.studio.playback.playback_controls():
                    break

    def _get_configs(self, user_configs, default_configs, valid_config_setup):
        config_mngr = ConfigManager(user_configs, default_configs, valid_config_setup)
        final = config_mngr.manage()
        return final["preprocessor"], final["estimator"]
    
    def _get_frame_names(self, view_style):
        view_style_names = {
            "inset": ["Original", "Threshold", "Edge Map"],
            "mosaic": ["Original", "Threshold", "Edge Map", "Composite"],
            "composite": ["Composite"]
        }
        try:
            names = view_style_names[view_style]
            return names
        except Exception as e:
            raise KeyError(f"ERROR: Invalid argument passed to 'view_style'. Must be one of {[key for key in view_style_names.keys()]}")
    
if __name__=="__main__":
    import numpy as np

    src = "../media/in/lane1-straight.mp4"
    # src = "../media/in/test_img1.jpg"

    roi = np.array([[[100, 540], 
                     [900, 540], 
                     [515, 320], 
                     [450, 320]]], dtype=np.int32)

    hough = CannyLaneDetector(src, roi)

    hough.detect("composite", stroke=True, fill=True)