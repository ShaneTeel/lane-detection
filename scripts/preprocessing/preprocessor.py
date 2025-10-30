import cv2
import numpy as np
from edge_map_generation import EdgeMapGenerator
from edge_point_extraction import EdgePointExtractor
from bev_transform import BEVTransformer


if __name__=="__main__":
    src1 = "../../media/in/test_img1.jpg"
    src2 = "../../media/in/lane1-straight.mp4"
    frame = cv2.imread(src1)
    cap = cv2.VideoCapture(src2)

    ret, frame = cap.read()
    cv2.imshow("Original", frame)
    cv2.waitKey(0)
    print(frame.shape)
    roi = np.array([[[450, 320], 
                     [515, 320], 
                     [900, 540], 
                     [100, 540]]], dtype=np.int32)
    
    canny = EdgeMapGenerator(roi)
    extractor = EdgePointExtractor(490)
    bev = BEVTransformer(roi)

    roi_mask, edge_map = canny.gen_edge_map(frame)
    cv2.imshow("Mask", roi_mask)
    cv2.waitKey(0)

    warped = bev._warp_frame(edge_map)
    print(edge_map.shape)
    # pts = extractor.extract_points(warped)

    cv2.imshow("Warped", warped)
    cv2.waitKey(0)

    cv2.destroyAllWindows()