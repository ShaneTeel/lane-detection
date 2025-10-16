    
import cv2
import numpy as np

class Illustrator:
    '''Superimposes shapes/lines on an image'''

    def __init__(self, stroke_color:tuple = (0, 0, 255), fill_color:tuple = (0, 255, 0)):
        
        self.stroke_color = self._hex_to_bgr(stroke_color)
        self.fill_color = self._hex_to_bgr(fill_color)

    def draw_curved_stroke_fill(self, frame, lines, stroke:bool=True, fill:bool=True, alpha:float=0.8, beta:float=0.3):
        if lines is None:
            raise ValueError("Error: argument passed fCannyKMeansor lines contains no lines.")
        if not stroke and not fill:
            assert AssertionError("ERROR: One of `stroke` or `fill` must be `True`. Both cannot be `False`.")

        canvas = np.zeros([frame.shape[0], frame.shape[1], 3], dtype=np.uint8)
        if stroke:
            self._draw_curved_lines(canvas, [lines[0]])
            self._draw_curved_lines(canvas, [lines[1]])
        if fill:
            self._draw_fill(canvas, lines)

        composite = cv2.addWeighted(frame, alpha, canvas, beta, 0.0)

        return composite
    
    def draw_straight_stroke_fill(self, frame, lines, stroke:bool=True, fill:bool=True, alpha:float=0.8, beta:float=0.3):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        if not stroke and not fill:
            assert AssertionError("ERROR: One of `stroke` or `fill` must be `True`. Both cannot be `False`.")
            
        canvas = np.zeros([frame.shape[0], frame.shape[1], 3], dtype=np.uint8)
        if stroke:
            self._draw_straight_lines(frame, [lines[0]])
            self._draw_straight_lines(frame, [lines[1]])
        if fill:
            points = np.array([[*[[x1, y1] for x1, y1, _, _ in lines[0]],
                                *[[x2, y2] for _, _, x2, y2 in lines[0]],
                                *[[x2, y2] for _, _, x2, y2 in lines[1]],
                                *[[x1, y1] for x1, y1, _, _ in lines[1]]]], dtype='int32')
            cv2.fillPoly(img=frame, pts=points, color=self.fill_color)
        return cv2.addWeighted(frame, alpha, canvas, beta, 0.0)

    def _draw_straight_lines(self, img, lines, color, thickness=1):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    
    def draw_polygon(self, frame, poly, stroke: bool = True, fill: bool = False, alpha:float=0.8, beta:float=0.3):
        
        pass

    def _draw_fill(self, frame, lines):
        if lines is None:
            raise ValueError("Lines are None")
            
        poly = np.concatenate(lines, dtype=np.int32)
        cv2.fillPoly(img=frame, pts=[poly], color=self.fill_color)

    def _draw_curved_lines(self, frame, points):
        if points is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        cv2.polylines(frame, points, isClosed=False, color=self.stroke_color, thickness=5, lineType=cv2.LINE_AA)

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