    
import cv2
import numpy as np

class Illustrator:
    '''Superimposes shapes/lines on an image'''

    def __init__(self, stroke_color:tuple = (0, 0, 255), fill_color:tuple = (0, 255, 0), alpha:float=0.8, beta:float=0.3):
        
        self.stroke_color = self._hex_to_bgr(stroke_color)
        self.fill_color = self._hex_to_bgr(fill_color)
        self.alpha = alpha
        self.beta = beta
    
    def _gen_hough_composite(self, frame, lines, stroke:bool=True, fill:bool=True):
        left, right = lines
        if left is None and right is None:
            print("No lines found, skipping")
            return frame
        if not stroke and not fill:
            assert AssertionError("ERROR: One of `stroke` or `fill` must be `True`. Both cannot be `False`.")
        canvas = np.zeros_like(frame)
        if stroke:
            self._draw_hough_lines(canvas, left)
            self._draw_hough_lines(canvas, right)
        if fill:
            if left is None or right is None:
                print("Left or right lane lines not found, skipping fill")
                return cv2.addWeighted(frame, self.alpha, canvas, self.beta, 0.0)
            self._draw_hough_fill(left, right, canvas)
        return cv2.addWeighted(frame, self.alpha, canvas, self.beta, 0.0)

    def _draw_hough_lines(self, img, line, color):
        if line is None:
            print("No lines, skipping.")
            return
        else:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

    def _draw_hough_fill(self, left, right, frame):
        poly = np.array([left[:2], left[2:], right[2:], right[:2]], dtype='int32').reshape(1, 4, 2)
        cv2.fillPoly(img=frame, pts=poly, color=self.fill_color)

    def _gen_ransac_composite(self, frame, lines, stroke:bool=True, fill:bool=True):
        if lines is None:
            raise ValueError("Error: argument passed fCannyKMeansor lines contains no lines.")
        if not stroke and not fill:
            assert AssertionError("ERROR: One of `stroke` or `fill` must be `True`. Both cannot be `False`.")

        canvas = np.zeros_like(frame)
        if stroke:
            self._draw_ransac_lines(canvas, [lines[0]])
            self._draw_ransac_lines(canvas, [lines[1]])
        if fill:
            self._draw_ransac_fill(canvas, lines)

        return cv2.addWeighted(frame, self.alpha, canvas, self.beta, 0.0)

    def _draw_ransac_fill(self, frame, lines):
        if lines is None:
            raise ValueError("Lines are None")
            
        poly = np.concatenate(lines, dtype=np.int32)
        cv2.fillPoly(img=frame, pts=[poly], color=self.fill_color)

    def _draw_ransac_lines(self, frame, points):
        if points is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        cv2.polylines(frame, points, isClosed=False, color=self.stroke_color, thickness=3, lineType=cv2.LINE_AA)

    def _draw_banner_text(self, frame, text):
        frame = self._channel_checker(frame)
        h, w = frame.shape[:2]
        banner_height = int(0.08 * h)
        cv2.rectangle(frame, (0, 0), (w, banner_height), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

        cv2.putText(frame, text, (int(w // 2) - 75, 10 + (banner_height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def _hex_to_bgr(self, color):
        if isinstance(color, tuple) and len(color) == 3:
            if len(color) == 3:
                return color
            else:
                return color[:3]
        
        if color.startswith("#"):
            hex_color = color[1:7]

            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            return (b, g, r)

    def _channel_checker(self, frame):
        if len(frame.shape) < 3:
            frame = cv2.merge([frame, frame, frame])
            return frame
        else:
            return frame