import cv2
import numpy as np
from typing import Literal
from utils import StudioManager, ConfigManager
from preprocessing import ROIManager, Preprocessor
from line_generators import StraightLineGenerator

class StraightLaneDetector():

    _VALID_HOUGH_SETUP = {
        "preprocessor": {
            'in_range': {'lower_bounds': [0, 255], 'upper_bounds': [0, 255]},
            'canny': {'weak_edge': [0, 301], 'sure_edge': [0, 301], 'blur_ksize': [3, 15], "blur_order": ["before", "after"]}
        },        
        "generator": {
            'hough': {'rho': None, 'theta': [1, 180], 'thresh': None, 'min_length': None, 'max_gap': None}
        }
    }

    _DEFAULT_HOUGH_CONFIGS = {
        "preprocessor": {
            'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
            'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "after"}
        },
        "estimator": {
            'hough': {'rho': 1, 'theta': 1, 'thresh': 50, 'min_length': 10, 'max_gap': 20},
            'linear': {}
        }
    }

    def __init__(self, source, roi:np.ndarray, configs:dict=None, stroke_color:tuple=(0, 0, 255), fill_color:tuple=(0, 255, 0)):
        if configs is None:
            pre_configs, gen_configs = self._DEFAULT_HOUGH_CONFIGS['preprocessor'], self._DEFAULT_HOUGH_CONFIGS["generator"]
        else:
            pre_configs, gen_configs = self._get_configs(configs, self._DEFAULT_HOUGH_CONFIGS, self._VALID_HOUGH_SETUP)

        self.roi = self._roi_validation(roi)
        self.preprocessor = Preprocessor(self.roi, pre_configs)
        self.generate = StraightLineGenerator(self.roi, gen_configs)
        self.studio = StudioManager(source, stroke_color, fill_color)
        
    def detect(self, view_style: Literal[None, "inset", "mosaic", "composite"]="inset", stroke:bool=False, fill:bool=True, save:bool = False):        
        win_name = f"{self.studio.source.name} {view_style} View"
        cv2.namedWindow(win_name)
        
        frame_names = self._get_frame_names(view_style)

        if self.studio.source.source_type != "image":
            self.studio.playback.print_playback_menu()
        
        if save:
            self.studio.write._initialize_writer()
        
        while True:
            ret, frame = self.studio.return_frame()
            if not ret:
                self.studio.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                roi_mask, edge_map = self.preprocessor.preprocess(frame)
                fit = self.generator.fit(edge_map)
                frame_lst = [frame, roi_mask, edge_map]
                final = self.studio.gen_view(frame_lst, frame_names, fit, view_style, stroke=stroke, fill=fill)

                cv2.imshow(win_name, final)

                if save:
                    self.studio.write.writer.write(final)

                if self.studio.playback.playback_controls():
                    break

    def _roi_validation(self, roi):
        if roi.shape == (1, 4, 2):
            return roi
        else:
            try:
                return roi.reshape(1, 4, 2)
            except Exception as e:
                raise ValueError(e)

    def _get_configs(self, user_configs, default_configs, valid_config_setup):
        config_mngr = ConfigManager(user_configs, default_configs, valid_config_setup)
        final = config_mngr.manage()
        return final["preprocessor"], final["generator"]
    
    def _get_frame_names(self, view_style):
        view_style_names = {
            "inset": ["Original", "Threshold", "Edge Map"],
            "mosaic": ["Original", "Threshold", "Edge Map", "Hough Composite"],
            "composite": ["Hough Composite"]
        }
        try:
            names = view_style_names[view_style]
            return names
        except Exception as e:
            raise KeyError(f"ERROR: Invalid argument passed to 'view_style'. Must be one of {[key for key in view_style_names.keys()]}")
    
if __name__=="__main__":

    # src = "../media/in/lane1-straight.mp4"
    src = "../media/in/test_img1.jpg"

    roi = np.array([[[100, 540], 
                     [900, 540], 
                     [515, 320], 
                     [450, 320]]], dtype=np.int32)

    hough = PPHTQuadraticLaneDetector(src, roi)

    hough.detect("composite", stroke=True, fill=True)