import cv2
import numpy.typing as npt
from studio import Reader, Writer, Render, Illustrator, Controller, Custodian
from detection import RANSACLineGenerator

class RANSACLaneDetector():

    def __init__(self, source, roi:npt.NDArray = None, configs:dict = None, stroke_color:tuple = (0, 0, 255), fill_color:tuple=(0, 255, 0)):

        self.roi = self._roi_validation(roi)
        self.processor = RANSACLineGenerator(self.roi, configs)
        self.source = Reader(source)
        self.writer = Writer(self.source)
        self.draw = Illustrator(stroke_color, fill_color)
        self.render = Render()
        self.controller = Controller(self.source)
        self.custodian = Custodian(self.source, self.writer)


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
            ret, frame = self.source.return_frame()
            if not ret:
                break
            else:
                thresh, roi_mask, edge_map, fit = self.processor.fit(frame)
                composite = self.draw.draw_curved_stroke_fill(frame, fit, stroke=False)
                final = self.render.render_final_view([frame, roi_mask, edge_map, composite], names)

                cv2.imshow("test", final)
                key = cv2.waitKey(1)
                if key == 27 or key == 32:
                    break

if __name__ == "__main__":
    import numpy as np

    source = "media/in/lane1-straight.mp4"
    # source = "media/in/test_img1.jpg"

    roi = np.array([[[100, 540], 
                     [900, 540], 
                     [515, 320], 
                     [450, 320]]])
        
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