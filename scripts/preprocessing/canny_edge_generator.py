import cv2

class CannyEdgeGenerator():
    _DEFAULT_CONFIGS = {
        'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
        'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "after"}
    }

    def __init__(self, configs:dict=None):

        if configs is None:
            configs = self._DEFAULT_CONFIGS
        self.lower_bounds = configs["in_range"].get("lower_bounds")
        self.upper_bounds = configs["in_range"].get("upper_bounds")
        self.weak_edge = configs["canny"].get("weak_edge")
        self.sure_edge = configs["canny"].get("sure_edge")
        self.blur_ksize = configs["canny"].get("blur_ksize")
        self.blur_order = configs["canny"].get("blur_order")
        
    def generate(self, frame):
        thresh = self._threshold(frame, self.lower_bounds, self.upper_bounds)
        edge_map = self._detect_edges(thresh, self.weak_edge, self.sure_edge, self.blur_ksize, self.blur_order)
        return thresh, edge_map
    
    def _threshold(self, frame, lower_bounds, upper_bounds):
        thresh = cv2.inRange(frame, lower_bounds, upper_bounds)
        return thresh

    def _detect_edges(self, roi, weak_edge, sure_edge, blur_ksize, blur_order):
        kernel = (blur_ksize, blur_ksize)
        if blur_order == 'before':
            img = cv2.GaussianBlur(roi, kernel, 0)
            img = cv2.Canny(img, weak_edge, sure_edge)
        else:
            img = cv2.Canny(roi, weak_edge, sure_edge)
            img = cv2.GaussianBlur(img, kernel, 0)
        return img
