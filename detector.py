import numpy as np

from handler import SourceHandler
from viewer import MediaViewer
from processor import ImageProcessor

class LaneDetector():

    _POLYGON = np.array([[[100, 540], [900, 540], [515, 320], [450, 320]]])
    _DEFAULT_CONFIG = {
        'processor': {
            'threshold': {'threshLow': 150, 'threshHigh': 255},
            'canny': {'cannyLow': 50, 'cannyHigh': 100, 'blurFirst': False},
            'hough': {'rho': 1, 'theta': np.radians(1), 'thresh': 50, 'minLength': 10, 'maxGap': 20},
            'composite': {'stroke': True, 'fill': True, 'poly': _POLYGON, 'alpha': 0.8, 'beta': 0.3, 'gamma': 0.0},
            'roi': {'poly': _POLYGON}
        },
        'viewer': {
            'maxWidth': 700
        },
        'roi': _POLYGON
    }

    def __init__(self, source, name=None, configs=None):
        
        if configs is None:
            configs = self._DEFAULT_CONFIG
        
        self.processorConfigs = configs['processor']
        self.viewerConfigs = configs['viewer']
        self.roi = configs['roi']
        self.source = SourceHandler(source, name)
        self.process = ImageProcessor(self.processorConfigs)
        self.view = MediaViewer(self.source, self.process, self.viewerConfigs)
    
    def previsImage(self):
        ret, frame = self.view.returnFrame()
        self.view.staticImage([frame], write=False)
    
    def postvisImage(self, compare:bool=False, write:bool=False):
        ret, raw = self.view.returnFrame()
        processed = self.process.imageProcessor(raw)
        if compare:
            self.view.staticImage([raw, processed], write)
        else:
            self.view.staticImage([processed], write)
        
    def previsVideo(self):
        self.view.videoStream(False, False, False)

    def postvisVideo(self, compare:bool=True, write:bool=False):
        self.view.videoStream(True, compare, write)

if __name__ == "__main__":
    
    file = './media/lane1-straight.mp4'
    video = LaneDetector(file, None, None)
    video.postvisVideo(compare=True, write=True)

    # file = './media/test_img1.jpg'
    # image = LaneDetector(file, None, None)
    # image.postvisImage(True, False)

    # source = SourceHandler('./lane1-straight-processed.mp4', None)
    # view = MediaViewer(source, None, None)
    # view.videoStream(False, False, False)