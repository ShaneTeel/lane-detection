if __name__=="__main__":
    import numpy as np
    from lane_detection.models import RANSACRegression, OLSRegression
    from lane_detection.preprocessing import CannyFeatureEngineer, HoughFeatureEngineer
    from lane_detection.detectors import BaseDetector
    
    src = "../media/in/lane1-straight.mp4"
    # src = "../media/in/test_img1.jpg"

    roi = np.array([[[75, 540], 
                     [925, 540], 
                     [520, 320], 
                     [450, 320]]], dtype=np.int32)
    
    estimator = RANSACRegression()
    preprocessor = CannyFeatureEngineer()

    detector = BaseDetector(src, preprocessor, estimator, roi)

    detector.detect("mosaic", stroke=True, fill=True)