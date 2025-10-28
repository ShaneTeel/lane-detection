import numpy as np
import cv2
from edge_detector import EdgeDetector
from feature_extraction import FindNonZero
from RANSAC_straight import StraightLineGenerator

if __name__=="__main__":
    # source = "../media/in/lane1-straight.mp4"
    source = "../../media/in/test_img1.jpg"

    roi = np.array([[[100, 540], 
                     [900, 540], 
                     [515, 320], 
                     [450, 320]]], dtype=np.int32)
    
    x_mid = roi[:, :, 0].mean()

    y_min = int(min(roi[0, 0, 1], roi[0, -1, 1]))
    y_max = int(max(roi[0, 0, 1], roi[0, -1, 1]))
        
    preprocessor = {
        'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
        'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "before"},
    }
    feature = {
        'filter': {'filter_type': 'median', 'n_std': 2.0}
    }

    frame = cv2.imread(source)

    canny = EdgeDetector(roi, preprocessor)

    extractor = FindNonZero(x_mid, "median", 2.0)

    gen = StraightLineGenerator(y_min, y_max)

    mask, edge_map = canny.preprocess(frame)
    pts = extractor.extract_points(edge_map)
    fit = gen.fit(pts)
    print(fit)