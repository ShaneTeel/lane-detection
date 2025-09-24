import cv2
import numpy as np
import math
import os

class LaneDetection():
    #--
    THRESHOLD_KWARGS = {'threshLow': 150, 'threshHigh': 255}

    POLYGON = np.array([[[100, 540],
                         [900, 540],
                         [515, 320],
                         [450, 320]]])

    CANNY_KWARGS = {'cannyLow': 50, 'cannyHigh': 100, 'blurFirst': False}

    HOUGH_KWARGS = {'rho': 1, 'theta': np.radians(1), 'thresh': 50, 'minLength': 10, 'maxGap': 20}

    COMPOSITE_KWARGS = {'stroke': True, 'fill': True, 'poly': POLYGON, 'alpha': 0.8, 'beta': 0.3, 'gamma': 0.0}

    ALL_KWARGS = {
        'poly': POLYGON,
        'thresholdParams': THRESHOLD_KWARGS,
        'cannyParams': CANNY_KWARGS,
        'houghParams': HOUGH_KWARGS,
        'compositeParams': COMPOSITE_KWARGS
    }

    def __init__(self, source, name):
        #--
        self.proceed = True
        self.source = source
        self.sourceType = None
        self.image = None
        self.cap = None
        self.writer = None
        self.fps = None
        self.frameCount = None
        self.fileOutName = None

        if name is None:
            if isinstance(self.source, int):
                self.name = f"camera_{source}"
                self.ext = ".mp4"
            else:
                self.name, self.ext = os.path.splitext(os.path.basename(self.source))
        else:
            self.name = name
            _, self.ext = os.path.splitext(os.path.basename(self.source))

        self._initializeSource()

    def _initializeSource(self):
        #--
        ##

        if isinstance(self.source, int):
            ##

            self._initializeCamera()
        ##

        elif isinstance(self.source, str):
            if self._isImageFile():
                ##

                self._initializeImage()
            else:
                ##

                self._initializeVideo()
        else:
            raise ValueError(f"Invalid source type: {type(self.source)}. Expected str or int.")

    def _isImageFile(self):
        validSuffix = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        _, ext = os.path.splitext(os.path.basename(self.source))
        return ext in validSuffix
    
    def _initializeImage(self):
        #--
        self.sourceType = 'image'
        self.image = cv2.imread(self.source)

        if self.image is None:
            raise ValueError(f"Error: Failed to read image from {self.source}")
        
        self.height, self.width = self.image.shape[:2]
        _, self.ext = os.path.splitext(self.source)
        print(f"Successfully read image: {self.source} ({self.height}x{self.width})")

    def _initializeVideo(self):
        #--
        self.sourceType = 'video'
        self.cap = cv2.VideoCapture(self.source)


        if not self.cap.isOpened():
            raise ValueError(f"Error: Failed to open video file {self.source}")
        
        else:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Successfully loaded video: {self.source} ({self.width}x{self.height}, {self.fps} FPS, {self.frameCount} frames)")

    def _initializeCamera(self):
        #--
        self.sourceType = 'camera'
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Failed to open camera file {self.source}")
        else:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Successfully opened camera: {self.source} ({self.width}x{self.height}, {self.fps:.1f} FPS)")

    def readFrame(self):
        #--
        if self.sourceType == 'image':
            return True, self.image.copy()
        
        else:
            if self.cap is None:
                return False, None
            
            ##

            ret, frame = self.cap.read()
            ##

            if ret:
                return True, frame
            else:
                return False, None
    
    def cleanUp(self):
        #--
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print(f"Released {self.sourceType} capture object.")
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Released {self.sourceType} writer object.")

    def __del__(self):
        #--
        self.cleanUp()
        print("Destroying all windows.")
        cv2.destroyAllWindows()
        print("\nThank you for using the Lane Detection module.")
        print("\n\t- Shane Teel")

    def runProcessor(self, write:bool=False, params=None):
        #--
        if params is None:
            params = self.ALL_KWARGS

        ret, frame = self.readFrame()

        if ret:
            if self.sourceType == 'video':
                self.videoProcessor(params)
            else:
                print(f"Processing {self.source}.")
                frame = self.imageProcessor(**params)
                print(f"Processing {self.source} complete.")
                self.showImages(self.image, frame, 600, False)
                self.cleanUp()
        else:
            raise ValueError(f"Failed to read {self.source}")

    def videoProcessor(self, write=False, videoParams=None):
        #--
        if videoParams is None:
            videoParams = self.ALL_KWARGS
        if not write:
            self.videoWriter()
        currentFrame = 0
        paused = False
        breakAll = False

        print(f"Processing {self.name}. Video contains {self.frameCount} total frames.")

        self._printPlaybackControls()

        while True and not breakAll:
            ret, frame = self.cap.read()
            if not ret:
                break

            else:
                self.image = frame
                frame = self.imageProcessor(**videoParams)
                
                if self.writer is not None:
                    self.writer.write(frame)

                cv2.imshow(self.name, frame)
                
                ##

                while paused:
                    key = cv2.waitKey(1) & 0xFF
                    if key & 0xFF in [ord('p'), ord(' ')]:
                        paused = False
                        print("Resuming video.")
                    elif key & 0xFF in [ord('q'), ord('Q'), 27]:
                        print(f"Pressed quit. Exiting video at frame {currentFrame}.")
                        breakAll = True
                        break

                ##

                if not paused:
                    key = cv2.waitKey(1) & 0xFF
                    if key & 0xFF in [ord('p'), ord(' ')]:
                        paused = True
                        print(f"Pausing video at frame {currentFrame}.")
                        print("Press 'p' or the spacebar to resume")
                        self._printPlaybackControls()
                    
                    elif key & 0xFF in [ord('q'), ord('Q'), 27]:
                        print(f"Exiting video at frame {currentFrame}.")
                        break

                    ##

                    ##

                    elif key & 0xFF == ord('r'):
                        print(f"Rewinding to frame {0 if currentFrame - 50 <= 0 else currentFrame - 50}")
                        currentFrame = max(0, currentFrame - 50)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)

                    ##

                    ##

                    elif key & 0xFF == ord('f'):
                        print(f"Skipping to frame {self.frameCount - 1 if currentFrame + 50 >= self.frameCount - 1 else currentFrame + 50}")
                        currentFrame = min(self.frameCount - 1, currentFrame + 50)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
                    
                    elif key & 0xFF == ord('h'):
                        self._printPlaybackControls()
                
                currentFrame += 1

        print(f"Processed {currentFrame + 1} of {self.frameCount} frames.")

        if self.writer is not None:
            print(f"Video saved to '{self.fileOutName}'")
        else:
            print("No video file saved.")

    def videoWriter(self):
        #--
        self.fileOutName = self.name + '-processed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.fileOutName, fourcc, self.fps, (self.width, self.height))

    def _printPlaybackControls(self):
        print("-----------------------------------------------")
        print("Use the following commands to control playback.\n")
        print("   Exit         : 'q' or ESC\n"              "   Pause/Resume : 'p' or SPACE\n"              "   Fast-Forward : 'f'\n"              "   Reverse      : 'r'\n"              "   Help         : 'h'")
        print("-----------------------------------------------")

    def imageProcessor(
            self,
            poly,
            thresholdParams=None,
            cannyParams=None,
            houghParams=None,
            compositeParams=None
            ):
        #--
        self.proceed = True
        ##

        if thresholdParams is None:
            thresholdParams = self.THRESHOLD_KWARGS
        threshold = self.thresholdLaneLines(**thresholdParams)

        ##

        if poly is None:
            poly = self.POLYGON
        roi = self.selectROI(threshold, poly)

        ##

        if cannyParams is None:
            cannyParams = self.CANNY_KWARGS
        edgeMap = self.detectEdges(roi, **cannyParams)

        ##

        if houghParams is None:
            houghParams = self.HOUGH_KWARGS
        _, lines = self.fitLines(edgeMap, **houghParams)

        ##

        if compositeParams is None:
            compositeParams = self.COMPOSITE_KWARGS
        combined = self.createCompositeImage(lines, **compositeParams)
        return combined
            
    def thresholdLaneLines(self, threshLow, threshHigh):
        #--
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.inRange(img, threshLow, threshHigh)
        return thresh

    def selectROI(self, threshImage, poly):
        #--        
        ##

        mask = np.zeros_like(threshImage)

        ##

        if len(threshImage.shape) > 2:
            numChannels = threshImage.shape[2]
            roiColor = (255,) * numChannels
        else:
            roiColor = 255
        
        ##

        cv2.fillPoly(img=mask, pts=poly, color=roiColor)

        ##

        roi = cv2.bitwise_and(src1=threshImage, src2=mask)
        return roi

    def detectEdges(self, roi, cannyLow, cannyHigh, blurFirst):
        #--
        if blurFirst == True:
            ##

            img = cv2.GaussianBlur(roi, (3, 3), 0)
            img = cv2.Canny(img, cannyLow, cannyHigh)
        else:
            img = cv2.Canny(roi, cannyLow, cannyHigh)
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def fitLines(self, edgeMap, rho, theta, thresh, minLength, maxGap):
        #--
        ##

        lines = cv2.HoughLinesP(edgeMap, rho, theta, thresh, minLength, maxGap)
        ##

        hough = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawLines(hough, lines)
        return hough, lines
    
    def createCompositeImage(self, lines, stroke, fill, poly, alpha, beta, gamma):
        #--
        ##

        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            lanesClassified = self.classifyLines(lines)
            laneLines = self.generateLineOfBestFit(lanesClassified, poly)
            if self.proceed == False:
                return self.image
            else:
                ##

                canvas = np.zeros([self.height, self.width, 3], dtype=np.uint8)
                self.drawStrokeFill(canvas, laneLines, stroke, fill)

                ##

                img = cv2.addWeighted(self.image, alpha, canvas, beta, gamma)
                return img
    
    def drawStrokeFill(self, img, lines, stroke, fill):
        #--
        ##

        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            if stroke:
                self.drawLines(img, [lines[0]], (0, 0, 255), 10)
                self.drawLines(img, [lines[1]], (0, 0, 255), 10)
            if fill:
                points = np.array([[*[[x1, y1] for x1, y1, _, _ in lines[0]],
                                    *[[x2, y2] for _, _, x2, y2 in lines[0]],
                                    *[[x2, y2] for _, _, x2, y2 in lines[1]],
                                    *[[x1, y1] for x1, y1, _, _ in lines[1]]]], dtype='int32')
                cv2.fillPoly(img=img, pts=points, color=(0, 255, 0))
    
    def generateLineOfBestFit(self, lanes, poly):
        #--
        ##

        yMin = min(poly[0][0][1], poly[0][-1][1])
        yMax = max(poly[0][0][1], poly[0][-1][1])

        for lateral in lanes.keys():
            lanes[lateral]['mAvg'] = self.calcAvg(lanes[lateral]['m'])
            lanes[lateral]['bAvg'] = self.calcAvg(lanes[lateral]['b'])

            if (math.isinf(lanes[lateral]['mAvg']) or math.isinf(lanes[lateral]['bAvg'])):
                self.proceed = False
                break
            else:
                ##

                xMin = int((yMin - lanes[lateral]['bAvg']) / lanes[lateral]['mAvg'])
                xMax = int((yMax - lanes[lateral]['bAvg']) / lanes[lateral]['mAvg'])

                ##

                lanes[lateral]['line'].append([xMin, yMin, xMax, yMax])
        return [lanes['left']['line'], lanes['right']['line']]

    def classifyLines(self, lines):
        #--
        ##

        lanes = {'left': {'m': [], 'b': [], 'mAvg': 0, 'bAvg': 0, 'line': []},
                'right': {'m': [], 'b': [], 'mAvg': 0, 'bAvg': 0, 'line': []}}
        for line in lines:
            m, b = self.calcSlopeIntercept(*line)
            if m is not None:
                if m < 0:
                    lanes['left']['m'].append(m)
                    lanes['left']['b'].append(b)
                elif m > 0:
                    lanes['right']['m'].append(m) 
                    lanes['right']['b'].append(b)
        return lanes
    

    def calcSlopeIntercept(self, line):
        #--
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

    def calcAvg(self, values):
        #--
        if values is None:
            raise ValueError(f"Error: {values} == 'NoneType'")
        else:
            if len(values) > 0:
                n = len(values)
            else:
                n = 1
            return sum(values) / n

    def drawLines(self, img, lines, color=(0, 0, 255), thickness=1):
        #--
        if lines is None:
            raise ValueError("Error: argument passed for lines contains no lines.")
        else:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    def showImages(self, img1, img2, maxWidth, write):
        #--
        ##

        h, w = img1.shape[:2]
        if w > maxWidth:
            ratio = maxWidth / w
            w = maxWidth
            h = int(ratio * h)
            left = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
            right = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        else:
            left = img1
            right = img2
        ##

        if len(left.shape) < 3:
            left = cv2.merge([left, left, left])
        
        if len(right.shape) < 3:
            right = cv2.merge([right, right, right])

        stacked = np.hstack([left, right])
        h, w = stacked.shape[:2]
        cv2.line(stacked, (w // 2, 0), (w // 2, h), (0, 0, 0), 1, cv2.LINE_AA)

        ##

        print(f"Displaying side-by-side single frame comparison of original image with processed image.")
        cv2.imshow(f"Original ({self.name}) vs Processed ({self.name})", stacked)
        print("Press any key to exit.")
        cv2.waitKey(0)

        ##

        if write:
            cv2.imwrite('Original-v-Overlay.jpg', stacked)

if __name__ == '__main__':
    ##


    file = './lane1-straight.mp4'

    ##


    img = LaneDetection(source=file, name=None)

    combined = img.runProcessor()