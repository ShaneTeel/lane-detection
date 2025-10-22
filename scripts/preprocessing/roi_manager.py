import numpy as np

class ROIManager():

    def _vanishing_point_roi(self):

        pass

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
        

        cv2.imshow("Edge Map", edge_map)
        cv2.waitKey(0)

        h, w = edge_map.shape[:2]
        temp = np.zeros_like(edge_map)
        result = np.zeros_like(edge_map)

        bottom = edge_map[h//2:, :]

        cv2.imshow("Bottom", bottom)
        cv2.waitKey(0)

        frame = cv2.medianBlur(bottom, 3)

        cv2.imshow("Blur", frame)
        cv2.waitKey(0)

        # for row in range(h // 2, h):
        #     row_hist = np.zeros(256, dtype=int)
        #     for kernel_row in range(5):
        #         for kernel_col in range():
        #             pixel = edge_map[row, col]
        #             row_hist[pixel] += 1
            
        #     temp[row, col] = np.average(row_hist)


        # cv2.imshow("Temp", temp)
        # cv2.waitKey(0)

        # for col in range(w):
        #     col_hist = np.zeros(256, dtype=int)
        #     for row in range(h//2, h):
        #         pixel = temp[row, col]
        #         col_hist[pixel] += 1
            
        #     result[row, col] = self._get_median(col_hist, h)

        # cv2.imshow("Result", result)
        # cv2.waitKey(0)

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
    
    def _get_median(self, hist, h):
        for intensity in range(256):
            if hist[intensity] > (h // 2) // 2:
                return 255
        return 0



    def _generate_edge_map(self, frame, weak_edge, sure_edge, blur_ksize, blur_order):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh = cv2.inRange(gray, 150, 255)

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

    test = ROIManager()

    roi = test._edge_density_roi(frame)

    mask = np.zeros_like(frame)

    cv2.fillPoly(img=mask, pts=[roi], color=(0, 255, 0))

    frame = cv2.addWeighted(frame, 0.8, mask, 0.15, 0.0)
    
    cv2.imshow("Test", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()