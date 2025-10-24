import cv2
import numpy as np

class Preprocessor:

    def __init__(self, roi:np.ndarray = None, configs:dict=None):
        self.roi = None if roi is None else roi
        self.in_range_params = configs["in_range"]
        self.canny_params = configs["canny"]

    def preprocess(self, frame):
        thresh = self._threshold_img(frame, **self.in_range_params)
        masked = self._inverse_roi_mask(thresh, self.roi)
        edge_map = self._generate_edge_map(masked, **self.canny_params)
        return masked, edge_map

    def _threshold_img(self, frame, lower_bounds, upper_bounds):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.inRange(img, lower_bounds, upper_bounds)
        return thresh

    def _inverse_roi_mask(self, thresh_img, poly):
        mask = np.zeros_like(thresh_img)
        if len(thresh_img.shape) > 2:
            num_channels = thresh_img.shape[2]
            roi_color = (255,) * num_channels
        else:
            roi_color = 255
        cv2.fillPoly(img=mask, pts=poly, color=roi_color)
        roi = cv2.bitwise_and(src1=thresh_img, src2=mask)
        return roi

    def _generate_edge_map(self, roi, weak_edge, sure_edge, blur_ksize, blur_order):
        kernel = (blur_ksize, blur_ksize)
        if blur_order == 'before':
            img = cv2.GaussianBlur(roi, kernel, 0)
            img = cv2.Canny(img, weak_edge, sure_edge)
        else:
            img = cv2.Canny(roi, weak_edge, sure_edge)
            img = cv2.GaussianBlur(img, kernel, 0)
        return img