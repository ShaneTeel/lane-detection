import cv2
import numpy as np

class BEVTransformer():

    def __init__(self, roi):
        self.src_pts = roi.astype(np.float32)
        self.dst_pts = np.array([[[25, 100],
                                  [515, 100],
                                  [515, 900],
                                  [25, 900]]],
                                  dtype=np.float32)
        
    def _warp_frame(self, frame):
        H = self._calc_matrix_H(self.src_pts, self.dst_pts)
        frame = cv2.warpPerspective(frame, H, (frame.shape))
        return frame
    
    def _calc_matrix_H(self, src_pts, dest_pts):

        A = []
        print(src_pts)
        for (xi, yi), (xj, yj) in zip(src_pts[0], dest_pts[0]):
            A.append([-xi, -yi, -1, 0, 0, 0, xi * xj, yi * yj, xj])
            A.append([0, 0, 0, -xi, -yi, -1, xi * yj, yi * yj, yj])

        A = np.array(A)
        U, S, V_T = np.linalg.svd(A)
        H = V_T[-1].reshape(3, 3)

        return H / H[2, 2]

    
# if __name__=="__main__":

#     src1 = "../../media/in/test_img1.jpg"
#     src2 = "../../media/in/lane1-straight.mp4"
#     frame = cv2.imread(src1)
#     cap = cv2.VideoCapture(src2)

#     ret, frame = cap.read()
#     print(frame.shape)
#     cv2.imshow("Original", frame)
#     cv2.waitKey(0)

#     roi = np.array([[[100, 540], 
#                      [900, 540], 
#                      [515, 320], 
#                      [450, 320]]], dtype=np.int32)
    

#     bev = BEVTransformer(roi, 540, 900)

#     warped = bev._warp_frame(frame)

#     cv2.imshow("Warped", warped)
#     cv2.waitKey(0)

#     cv2.destroyAllWindows()

