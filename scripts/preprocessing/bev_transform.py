import cv2
import numpy as np

class BEVTransformer():

    def __init__(self, roi, size:tuple, x_max, y_max):
    
        self.src_pts = roi.astype(np.float32)
        self.h, self.w = size

        src_bottom_left, src_bottom_right, src_top_right, src_top_left = self.src_pts[0]
        bottom_width = np.linalg.norm(src_bottom_right - src_bottom_left)
        roi_height = np.linalg.norm(src_bottom_left - src_top_left)
        dst_width = int(bottom_width)
        dst_height = int(roi_height * 1.5)

        bottom_left = [0, dst_height]
        bottom_right = [dst_width, dst_height]
        top_right = [dst_width, 0]
        top_left = [0, 0]
        
        self.dst_pts = np.array([[bottom_left,
                                  bottom_right,
                                  top_right,
                                  top_left]],
                                  dtype=np.float32)
        
        self.H = self._calc_H_mat()
        self.H_I = np.linalg.inv(self.H)
        
    def transform(self, frame):
        return cv2.warpPerspective(frame, self.H, (self.w, self.h), flags=cv2.INTER_NEAREST)
    
    # def transform(self, pts):
    #     return cv2.perspectiveTransform(pts, self.H)

    def inverse_transform(self, pts):
        pts = cv2.perspectiveTransform(pts, self.H_I)
        return pts.astype(np.int32)
    
    def _calc_H_mat(self):

        A = np.zeros((9, 9), dtype=np.float32)
        A[8, 8] = 1

        xi_yi = self.src_pts[0]
        ui_vi = self.dst_pts[0]
        DOF = list(range(0, 8, 2))

        for dof, (xi, yi), (ui, vi) in zip(DOF, xi_yi, ui_vi):
            A[dof,:] = np.array([-xi, -yi, -1, 0, 0, 0, xi * ui, yi * ui, ui])
            A[dof+1,:] = np.array([0, 0, 0, -xi, -yi, -1, xi * vi, yi * vi, vi])

        b = np.array([0]*8 + [1], dtype=np.float32)

        H = np.linalg.solve(A, b).reshape(3, 3)

        return H / H[2, 2]