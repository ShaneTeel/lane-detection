import cv2
import numpy as np
import math

class ImageProcessor():
    '''Test'''
    _POLYGON = np.array([[[100, 540], [900, 540], [515, 320], [450, 320]]])
    _DEFAULT_CONFIG = {
                'threshold': {'threshLow': 150, 'threshHigh': 255},
                'canny': {'cannyLow': 50, 'cannyHigh': 100, 'blurFirst': False},
                'hough': {'rho': 1, 'theta': np.radians(1), 'thresh': 50, 'minLength': 10, 'maxGap': 20},
                'composite': {'stroke': True, 'fill': True, 'poly': _POLYGON, 'alpha': 0.8, 'beta': 0.3, 'gamma': 0.0},
                'roi': {'poly': _POLYGON}
            }

    def __init__(self, processorConfig):
        if processorConfig is None:
            processorConfig = self._DEFAULT_CONFIG
    
        self.thresholdParams = processorConfig['threshold']
        self.cannyParams = processorConfig['canny']
        self.houghParams = processorConfig['hough']
        self.compositeParams = processorConfig['composite']
        self.roi = processorConfig['roi']
        self._validateConfigs() # Validate configs
    
    def _validateConfigs(self):
        attributes = [self.thresholdParams, self.cannyParams, self.houghParams, self.compositeParams, self.roi]
        for i, step in enumerate(self._DEFAULT_CONFIG.keys()):
            parameters = [val for val in self._DEFAULT_CONFIG[step]]
            for param in parameters:
                    if param not in attributes[i]:
                        raise ValueError(f"Missing required threshold parameter: {param}")
                    
        ####################################################
        # ADD VALIDATION FOR PARAMETER VALUE TYPE / RANGES #
        ####################################################

    def imageProcessor(self, frame):
        '''ADD LATER'''
        threshold = self._thresholdLaneLines(frame, **self.thresholdParams)
        roi = self._selectROI(threshold, **self.roi)
        edgeMap = self._detectEdges(roi, **self.cannyParams)
        _, lines = self._fitLines(frame, edgeMap, **self.houghParams)
        return self._createCompositeImage(frame, lines, **self.compositeParams)
            
    def _thresholdLaneLines(self, frame, threshLow, threshHigh):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.inRange(img, threshLow, threshHigh)
        return thresh

    def _selectROI(self, threshImage, poly):
        mask = np.zeros_like(threshImage)
        if len(threshImage.shape) > 2:
            numChannels = threshImage.shape[2]
            roiColor = (255,) * numChannels
        else:
            roiColor = 255
        cv2.fillPoly(img=mask, pts=poly, color=roiColor)
        roi = cv2.bitwise_and(src1=threshImage, src2=mask)
        return roi

    def _detectEdges(self, roi, cannyLow, cannyHigh, blurFirst):
        if blurFirst == True:
            img = cv2.GaussianBlur(roi, (3, 3), 0)
            img = cv2.Canny(img, cannyLow, cannyHigh)
        else:
            img = cv2.Canny(roi, cannyLow, cannyHigh)
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def _fitLines(self, frame, edgeMap, rho, theta, thresh, minLength, maxGap):
        lines = cv2.HoughLinesP(edgeMap, rho, theta, thresh, minLength, maxGap)
        hough = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        self._drawLines(hough, lines)
        return hough, lines
    
    def _createCompositeImage(self, frame ,lines, stroke, fill, poly, alpha, beta, gamma):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            lanesClassified = self._classifyLines(lines)
            laneLines = self._generateLineOfBestFit(lanesClassified, poly)

            canvas = np.zeros([frame.shape[0], frame.shape[1], 3], dtype=np.uint8)
            self._drawStrokeFill(canvas, laneLines, stroke, fill)

            img = cv2.addWeighted(frame, alpha, canvas, beta, gamma)
            return img
    
    def _drawStrokeFill(self, img, lines, stroke, fill):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            if stroke:
                self._drawLines(img, [lines[0]], (0, 0, 255), 10)
                self._drawLines(img, [lines[1]], (0, 0, 255), 10)
            if fill:
                points = np.array([[*[[x1, y1] for x1, y1, _, _ in lines[0]],
                                    *[[x2, y2] for _, _, x2, y2 in lines[0]],
                                    *[[x2, y2] for _, _, x2, y2 in lines[1]],
                                    *[[x1, y1] for x1, y1, _, _ in lines[1]]]], dtype='int32')
                cv2.fillPoly(img=img, pts=points, color=(0, 255, 0))
    
    def _generateLineOfBestFit(self, lanes, poly):
        yMin = min(poly[0][0][1], poly[0][-1][1])
        yMax = max(poly[0][0][1], poly[0][-1][1])

        for lateral in lanes.keys():
            lanes[lateral]['mAvg'] = self._calcAvg(lanes[lateral]['m'])
            lanes[lateral]['bAvg'] = self._calcAvg(lanes[lateral]['b'])

            if (math.isinf(lanes[lateral]['mAvg']) or math.isinf(lanes[lateral]['bAvg'])):
                raise ValueError("Error: Infinite float.")
            else:
                xMin = int((yMin - lanes[lateral]['bAvg']) / lanes[lateral]['mAvg'])
                xMax = int((yMax - lanes[lateral]['bAvg']) / lanes[lateral]['mAvg'])

                lanes[lateral]['line'].append([xMin, yMin, xMax, yMax])
        return [lanes['left']['line'], lanes['right']['line']]

    def _classifyLines(self, lines):
        lanes = {'left': {'m': [], 'b': [], 'mAvg': 0, 'bAvg': 0, 'line': []},
                'right': {'m': [], 'b': [], 'mAvg': 0, 'bAvg': 0, 'line': []}}
        for line in lines:
            m, b = self._calcSlopeIntercept(*line)
            if m is not None:
                if m < 0:
                    lanes['left']['m'].append(m)
                    lanes['left']['b'].append(b)
                elif m > 0:
                    lanes['right']['m'].append(m) 
                    lanes['right']['b'].append(b)
        return lanes
    

    def _calcSlopeIntercept(self, line):
        if line is None:
            raise ValueError(f"Error: {line} == 'NoneType'")
        else:
            x1, y1, x2, y2 = line
            if x1 == x2:
                print(f"Warning: Vertical line detected {line}, skipping")
                return None, None
            
            m = (y1 - y2) / (x1 - x2)
            b = y1 - (m * x1)
            return m, b

    def _calcAvg(self, values):
        if values is None:
            raise ValueError(f"Error: {values} == 'NoneType'")
        else:
            if len(values) > 0:
                n = len(values)
            else:
                n = 1
            return sum(values) / n

    def _drawLines(self, img, lines, color=(0, 0, 255), thickness=1):
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
