import cv2
import numpy as np
import math
from sklearn.cluster import KMeans

class CannyKMeans():
    '''Test'''
    _POLYGON = np.array([[[100, 540], [900, 540], [515, 320], [450, 320]]])
    _DEFAULT_CONFIG = {
                'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
                'canny': {'canny_low': 50, 'canny_high': 100, 'blur_first': False},
                'composite': {'stroke': True, "stroke_color": (0, 0, 255), 'fill': True, "fill_color": (0, 255, 0)}
            }

    def __init__(self, roi = None, configs = None):
        if configs is None:
            configs = self._DEFAULT_CONFIG
        
        if roi is not None:
            roi = np.array([[pt for pt in roi]])
        else:
            roi = self._POLYGON
        self.in_range_params = configs['in_range']
        self.canny_params = configs['canny']
        self.composite_params = configs['composite']
        self.roi = roi
        self.kmeans = KMeans(2, random_state=42)
        self._validate_configs() # Validate configs
        self._hex_to_bgr('fill_color')
        self._hex_to_bgr('stroke_color')

    def _validate_configs(self):
        attributes = [self.in_range_params, self.canny_params, self.composite_params]
        for i, step in enumerate(self._DEFAULT_CONFIG.keys()):
            parameters = [val for val in self._DEFAULT_CONFIG[step]]
            for param in parameters:
                    if param not in attributes[i]:
                        raise ValueError(f"Missing required threshold parameter: {param}")


        ####################################################
        # ADD VALIDATION FOR PARAMETER VALUE TYPE / RANGES #
        ####################################################

    def _hex_to_bgr(self, key):
        hex_color = self.composite_params.get(key)
        if isinstance(hex_color, tuple):
            self.composite_params[key] = hex_color
            return
        elif hex_color.startswith("#"):
            hex_color = hex_color[1:]
        
        if len(hex_color) != 6:
            raise ValueError("Invalid hex color code format. Expected six characters (remove alpha channel)")
        
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        self.composite_params[key] = (b, g, r)


    def run(self, frame):
        '''ADD LATER'''
        threshold = self._threshold_lane_lines(frame, **self.in_range_params)
        roi = self._select_ROI(threshold, self.roi)
        edge_map = self._detect_edges(roi, **self.canny_params)
        composite = self._create_composite(frame, edge_map, self._POLYGON, **self.composite_params)
        return composite

    def _threshold_lane_lines(self, frame, lower_bounds, upper_bounds):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.inRange(gray, lower_bounds, upper_bounds)
        return thresh

    def _select_ROI(self, thresh_img, poly):
        mask = np.zeros_like(thresh_img)
        if len(thresh_img.shape) > 2:
            num_channels = thresh_img.shape[2]
            roi_color = (255,) * num_channels
        else:
            roi_color = 255
        cv2.fillPoly(img=mask, pts=poly, color=roi_color)
        roi = cv2.bitwise_and(src1=thresh_img, src2=mask)
        return roi

    def _detect_edges(self, roi, canny_low, canny_high, blur_first):
        if blur_first == True:
            img = cv2.GaussianBlur(roi, (3, 3), 0)
            img = cv2.Canny(img, canny_low, canny_high)
        else:
            img = cv2.Canny(roi, canny_low, canny_high)
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def _create_composite(self, frame, edge_map, poly, stroke, stroke_color, fill, fill_color):
        if edge_map is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            pts = self._point_extraction(edge_map)
            clustered = self._point_clustering(pts)
            fit = [self._fit_poly(lane) for lane in clustered]

            canvas = np.zeros([frame.shape[0], frame.shape[1], 3], dtype=np.uint8)
            self._draw_stroke_fill(canvas, fit, stroke, stroke_color, fill, fill_color)

            composite = cv2.addWeighted(frame, 0.8, canvas, 0.3, 0.0)
            return composite
    
    def _point_extraction(self, edge_map):
        edge_pts = np.where(edge_map != 0)
        return np.column_stack((edge_pts[1], edge_pts[0]))
    
    def _point_clustering(self, pts):
        labels = self.kmeans.fit_predict(pts)

        left = pts[labels == 0]
        right = pts[labels == 1]
        return [left, right]

    def _fit_poly(self, lane):
        x = np.array([x for x, _ in lane])
        y = np.array([y for _, y in lane])

        coeff = np.polyfit(x, y, deg=2)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = np.polyval(coeff, x_smooth)

        points = np.array([x_smooth, y_smooth], dtype=np.int32).T
        return points.reshape((-1, 1, 2))
    
    def _draw_stroke_fill(self, canvas, lanes, stroke, stroke_color, fill, fill_color):
        if lanes is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            if stroke:
                self._draw_lines(canvas, [lanes[0]], stroke_color, 10)
                self._draw_lines(canvas, [lanes[1]], stroke_color, 10)
            if fill:
                self._draw_fill(canvas, lanes, fill_color)

    def _draw_fill(self, frame, lines, color):
        if lines is None:
            raise ValueError("Lines are None")
            
        poly = np.concatenate(lines, dtype=np.int32)
        cv2.fillPoly(img=frame, pts=[poly], color=color)

    def _draw_lines(self, frame, points, color, thickness):
        if points is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            cv2.polylines(frame, points, isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    
    def _gen_line_of_best_fit(self, lanes, poly):
        yMin = min(poly[0][0][1], poly[0][-1][1])
        yMax = max(poly[0][0][1], poly[0][-1][1])

        for lateral in lanes.keys():
            lanes[lateral]['mAvg'] = self._calc_avg(lanes[lateral]['m'])
            lanes[lateral]['bAvg'] = self._calc_avg(lanes[lateral]['b'])

            if math.isinf(lanes[lateral]['mAvg']) or math.isinf(lanes[lateral]['bAvg']):
                raise ValueError("Error: Infinite float.")
            else:
                try:
                    xMin = int((yMin - lanes[lateral]['bAvg']) // lanes[lateral]['mAvg'])
                    xMax = int((yMax - lanes[lateral]['bAvg']) // lanes[lateral]['mAvg'])
                    lanes[lateral]['line'].append([xMin, yMin, xMax, yMax])
                except Exception as e:
                    raise ValueError(f"Error: {e}")
                
        return [lanes['left']['line'], lanes['right']['line']]

if __name__ == "__main__":

    cap = cv2.VideoCapture("../../media/lane1-straight.mp4")

    if not cap.isOpened():
        print("Error: Could not open video.")

    processor = CannyKMeans()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            composite = processor.run(frame)
            cv2.imshow("test", composite)
            cv2.waitKey(1)
    
    cv2.destroyAllWindows()
