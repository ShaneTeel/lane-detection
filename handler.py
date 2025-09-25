import cv2
import os

class SourceHandler():

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.sourceType = None
        self.image = None
        self.cap = None
        self.writer = None
        self.width = None
        self.height = None
        self.fps = None
        self.frameCount = None

        self._initializeSource()

    def _initializeSource(self):
        '''ADD'''
        if isinstance(self.source, int):
            self._initializeCamera()
        elif isinstance(self.source, str):
            if self._isImageFile():
                self._initializeImage()
            else:
                self._initializeVideo()
        else:
            raise ValueError(f"Invalid source type: {type(self.source)}. Expected str or int.")

    def _isImageFile(self):
        '''ADD'''
        validSuffix = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        _, ext = os.path.splitext(os.path.basename(self.source))
        return ext in validSuffix
    
    def _initializeImage(self):
        '''ADD'''
        self.sourceType = 'image'
        self.image = cv2.imread(self.source)

        if self.image is None:
            raise ValueError(f"Error: Failed to read image from {self.source}")
        
        self.height, self.width = self.image.shape[:2]
        
        _, self.ext = os.path.splitext(os.path.basename(self.source))
        if self.name is None:
            self.name, _ = os.path.splitext(os.path.basename(self.source))

        print(f"Successfully read image {self.name}: {self.source} ({self.height}x{self.width})")

    def _initializeVideo(self):
        '''ADD'''
        self.sourceType = 'video'
        self.cap = cv2.VideoCapture(self.source)


        if not self.cap.isOpened():
            raise ValueError(f"Error: Failed to open video file {self.source}")
        
        else:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            _, self.ext = os.path.splitext(os.path.basename(self.source))
            if self.name is None:
                self.name, _ = os.path.splitext(os.path.basename(self.source))
            
            print(f"Successfully loaded video: {self.source} ({self.width}x{self.height}, {self.fps} FPS, {self.frameCount} frames)")

    def _initializeCamera(self):
        '''ADD'''
        self.sourceType = 'camera'
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Failed to open camera file {self.source}")
        else:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            if self.name is None:
                self.name = 'camera_' + str(self.source)
            
            print(f"Successfully opened camera: {self.source} ({self.width}x{self.height}, {self.fps:.1f} FPS)")