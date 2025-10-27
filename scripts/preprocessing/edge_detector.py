import cv2
import numpy as np

class EdgeDetector():

    _VALID_SETUP = {
            'in_range': {'lower_bounds': [0, 255], 'upper_bounds': [0, 255]},
            'canny': {'weak_edge': [0, 301], 'sure_edge': [0, 301], 'blur_ksize': [3, 15], "blur_order": ["before", "after"]}
        }

    _DEFAULT_CONFIGS = {
            'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
            'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "after"}
        }
    
    def __init__(self, roi:np.ndarray = None, configs:dict=None):
        if configs is None:
            configs = self._DEFAULT_CONFIGS

        self.roi = None if roi is None else roi
        self.lower_bounds = configs["in_range"].get("lower_bounds")
        self.upper_bounds = configs["in_range"].get("upper_bounds")
        self.weak_edge = configs["canny"].get("weak_edge")
        self.sure_edge = configs["canny"].get("sure_edge")
        self.blur_ksize = configs["canny"].get("blur_ksize")
        self.blur_order = configs["canny"].get("blur_order")

    def preprocess(self, frame):
        thresh = self._threshold_img(frame, self.lower_bounds, self.upper_bounds)
        masked = self._inverse_roi_mask(thresh, self.roi)
        edge_map = self._generate_edge_map(masked, self.weak_edge, self.sure_edge, self.blur_ksize, self.blur_order)
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