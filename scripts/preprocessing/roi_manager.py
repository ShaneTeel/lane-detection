import numpy as np

class ROIManager():

    def _vanishing_point_roi(self):

        pass

    def _simple_roi(self, frame):
        
        h, w = frame.shape[:2]

        roi = [[int(w * 0.15), h],
               [int(w * 0.85), h],
               [int(w * 0.55), int(h * 0.60)],
               [int(w * 0.45), int(h * 0.60)]
        ]
        return np.array([roi])
    
    def _bootstrap_simple_roi(self, frame):

        pass

    def _edge_density_roi(self, frame, percentile:int = 50):
        edge_map = self._generate_edge_map(frame, 50, 100, 3, "before")
        
        h, w = frame.shape[:2]

        bottom = edge_map[h//2:, :]
        
        bottom = edge_map[:, w//2]

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

    def _generate_edge_map(self, frame, weak_edge, sure_edge, blur_ksize, blur_order):
        kernel = (blur_ksize, blur_ksize)
        if blur_order == 'before':
            img = cv2.GaussianBlur(frame, kernel, 0)
            img = cv2.Canny(img, weak_edge, sure_edge)
        else:
            img = cv2.Canny(frame, weak_edge, sure_edge)
            img = cv2.GaussianBlur(img, kernel, 0)
        return img

    

if __name__ == "__main__":
    import cv2

    frame = cv2.imread('media/in/test_img1.jpg')

    test = ROIManager()

    roi = test._edge_density_roi(frame, 75)

    mask = np.zeros_like(frame)

    cv2.fillPoly(img=mask, pts=[roi], color=(0, 255, 0))

    frame = cv2.addWeighted(frame, 0.8, mask, 0.15, 0.0)
    
    cv2.imshow("Test", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()