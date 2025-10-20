import cv2
import numpy as np
import numpy.typing as npt
from utils import StudioManager, ConfigManager
from preprocessing import Preprocessor
from line_generators import RANSACLineGenerator

class RANSACLaneDetector():

    _VALID_RANSAC_SETUP = {
        "preprocessor": {
            'in_range': {'lower_bounds': list(range(0, 255)), 'upper_bounds': list(range(1, 256))},
            'canny': {'weak_edge': list(range(0, 301)), 'sure_edge': list(range(0, 301)), 'blur_ksize': list(range(3, 16, 2)), "blur_order": ["before", "after"]},
        },        
        "generator": {
            'filter': {'filter_type': ['median', 'mean'], 'n_std': np.arange(0.0, 3.1, 0.01).tolist()},
            'polyfit': {'n_iter': list(range(1, 101)), 'degree': [1, 2, 3], 'threshold': list(range(0, 101)), 'min_inliers': np.arange(0.0, 1.00, 0.01), 'weight': list(range(1, 11)), "factor": np.arange(0.0, 1.00, 0.01)},
        }
    }

    _DEFAULT_RANSAC_CONFIGS = {
        "preprocessor": {
            'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
            'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "before"},
        },
        "generator": {
            'filter': {'filter_type': 'median', 'n_std': 2}, 
            'polyfit': {'n_iter': 100, 'degree': 2, 'threshold': 50, 'min_inliers': 0.6, 'weight': 5, 'factor': 0.1}
        }
    }

    def __init__(self, source, roi:npt.NDArray = None, configs:dict = None, stroke_color:tuple = (0, 0, 255), fill_color:tuple=(0, 255, 0)):

        if configs is None:
            pre_configs, gen_configs = self._DEFAULT_RANSAC_CONFIGS['preprocessor'], self._DEFAULT_RANSAC_CONFIGS['generator']
        
        else:
            pre_configs, gen_configs = self._get_configs(configs, self._DEFAULT_RANSAC_CONFIGS, self._VALID_RANSAC_SETUP)

        self.roi = self._roi_validation(roi)
        self.preprocess = Preprocessor(self.roi, pre_configs)
        self.generate = RANSACLineGenerator(self.roi, gen_configs)
        self.studio = StudioManager(source, stroke_color, fill_color)

    def _roi_validation(self, roi):
        expected_shape = (1, 4, 2)
        if roi.shape == expected_shape:
            return roi
        else:
            try:
                return roi.reshape(1, 4, 2)
            except Exception as e:
                raise ValueError(e)

    def detect(self, names:list = None):
        if names is None:
            names = ["Raw", "Thresh", "Edge", "Composite"]
        while True:
            ret, frame = self.studio.source.return_frame()
            if not ret:
                break
            else:
                _, roi_mask, edge_map = self.preprocess.preprocess(frame)
                edge_map, fit = self.generate.fit(edge_map)
                composite = self.studio.draw.draw_curved_stroke_fill(frame, fit, stroke=False)
                final = self.studio.render.render_final_view([frame, roi_mask, edge_map, composite], names)

                cv2.imshow("test", final)
                key = cv2.waitKey(1)
                if key == 27 or key == 32:
                    break

    def _get_configs(self, user_configs, default_configs, valid_config_setup):
        config_mngr = ConfigManager(user_configs=user_configs, default_configs=default_configs, valid_config_setup=valid_config_setup)
        final = config_mngr.manage()
        return final['preprocessor'], final['generator']

if __name__ == "__main__":

    source = "media/in/lane1-straight.mp4"
    # source = "media/in/test_img1.jpg"

    roi = np.array([[[100, 540], 
                     [900, 540], 
                     [515, 320], 
                     [450, 320]]], dtype=np.int32)
        
    user_configs = {
        "preprocessor": {
            'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
            'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "before"},
        },
        "generator": {
            'filter': {'filter_type': 'median', 'n_std': 2.0},
            'polyfit': {'n_iter': 100, 'degree': 2, 'threshold': 50, 'min_inliers': 0.6, 'weight': 5, 'factor': 0.1}
        }
    }

    detector = RANSACLaneDetector(source, roi, user_configs)

    detector.detect()


"""
Tasks:
	# 1. Config manager
	2. ROI Calculator
	3. Standardize RANSAC and HOUGHP
	4. Find a way to add the intermediate step images to the final image (top row)
	5. Hyperparameter Tuning
	6. Batch Processing
    7. Clean-up RANSAC lines (they are innacurate with a great amount of variance)
        7a. Generate lines that reach the min/max of the selected/calculated ROI
"""