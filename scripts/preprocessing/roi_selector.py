import numpy as np
from scipy.signal import find_peaks

class ROISelector():

    def __init__(self, roi:np.ndarray=None):
        self.roi = self._roi_validation(roi)
        self.x_mid = None
        self.y_max = None
        self.y_min = None

        self._roi_extraction()

    def _roi_extraction(self):
        self.x_mid = self.roi[:, :, 0].mean()
        self.y_min = int(min(self.roi[0, 0, 1], self.roi[0, -1, 1]))
        self.y_max = int(max(self.roi[0, 0, 1], self.roi[0, -1, 1]))

    def _vanishing_point_roi(self):

        pass

    def _roi_validation(self, roi):
        if roi.shape == (1, 4, 2):
            return roi
        else:
            try:
                return roi.reshape(1, 4, 2)
            except Exception as e:
                raise ValueError(e)

    def _roi_heuristics(self, frame):
        
        h, w = frame.shape[:2]

        roi = [[int(w * 0.05), h],
               [int(w * 0.95), h],
               [int(w * 0.55), int(h * 0.60)],
               [int(w * 0.45), int(h * 0.60)]
        ]
        return np.array([roi])
    
    def _bootstrap_roi_heuristics(self, frame):

        pass

    def _edge_density_roi(self, frame, percentile:int = 75):
        edge_map = self._generate_edge_map(frame, 50, 100, 3, "before")
                
        h, w = edge_map.shape[:2]
        bottom = edge_map[h//2:, :]

        density =  np.sum(bottom, axis=0)
        thresh = np.percentile(density, percentile)

        pts = np.where(density > thresh)[0]

        bottom_left = pts.min()
        bottom_right = pts.max()

        roi = np.array([[[bottom_left, h],
                         [bottom_right, h],
                         [int(w * 0.55), int(h * 0.60)],
                         [int(w * 0.45), int(h * 0.60)]]], dtype=np.int32)
        return roi
    
    def _edge_density_roi2(self, frame):
        edge_map = self._generate_edge_map(frame, 50, 150, 3, 'before')
        
        h, w = edge_map.shape
        bottom = edge_map[h//2:, :]
        density = np.sum(bottom, axis=0)
        peaks, _ = find_peaks(density, distance=100, prominence=50)

    def _generate_edge_map(self, frame, weak_edge, sure_edge, blur_ksize, blur_order):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh = cv2.inRange(gray, 200, 255)

        kernel = (blur_ksize, blur_ksize)

        if blur_order == 'before':
            img = cv2.GaussianBlur(thresh, kernel, 0)
            img = cv2.Canny(img, weak_edge, sure_edge)
        else:
            img = cv2.Canny(thresh, weak_edge, sure_edge)
            img = cv2.GaussianBlur(img, kernel, 0)
        return img
    


if __name__ == "__main__":
    import cv2

    frame = cv2.imread('media/in/test_img1.jpg')
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])

    test = ROISegmentor()

    roi = test._edge_density_roi(frame)

    mask = np.zeros_like(frame)

    cv2.fillPoly(img=mask, pts=[roi], color=(0, 255, 0))

    frame = cv2.addWeighted(frame, 0.8, mask, 0.15, 0.0)
    
    cv2.imshow("Test", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()