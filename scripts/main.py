if __name__=="__main__":
    import numpy as np
    from lane_detection.models import RANSACRegression, OLSRegression, KalmanFilteredRANSAC, KalmanFilteredOLS
    from lane_detection.preprocessing import CannyFeatureEngineer, HoughFeatureEngineer
    from lane_detection.detectors import BaseDetector
    
    src = "../media/in/lane1-straight.mp4"
    # src = "../media/in/test_img1.jpg"

    roi = np.array([[[75, 540], 
                     [925, 540], 
                     [520, 320], 
                     [450, 320]]], dtype=np.int32)
    
    estimator = KalmanFilteredRANSAC(min_inliers=0.7, max_error=10)
    preprocessor = HoughFeatureEngineer()

    detector = BaseDetector(src, preprocessor, estimator, roi)

    detector.detect("composite", stroke=True, fill=True)