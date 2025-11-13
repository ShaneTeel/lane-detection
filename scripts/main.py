if __name__=="__main__":
    import numpy as np
    from lane_detection.detector import KalmanOLSLaneDetector, KalmanRANSACLaneDetector, OLSLaneDetector, RANSACLaneDetector 
    from lane_detection.preprocessing import CannyPreprocessor, HoughPreprocessor
    from lane_detection.pipeline import LaneDetectionSystem
    
    src = "../media/in/lane1-straight.mp4"
    # src = "../media/in/test_img1.jpg"

    roi = np.array([[[75, 540], 
                     [925, 540], 
                     [520, 320], 
                     [450, 320]]], dtype=np.int32)
    
    left_lane = OLSLaneDetector()
    right_lane = OLSLaneDetector()
    preprocessor = CannyPreprocessor()

    system = LaneDetectionSystem(src, roi, preprocessor, left_lane, right_lane)

    eval = system.run("composite", stroke=True, fill=True)
    
    print(eval)
