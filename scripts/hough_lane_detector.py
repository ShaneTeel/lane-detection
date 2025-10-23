import cv2
import numpy as np
from utils import StudioManager, ConfigManager
from preprocessing import Preprocessor
from line_generators import HoughLineGenerator

class HoughLaneDetector():
    
    def __init__(self, source, roi:np.ndarray, configs:dict, stroke_color:tuple=(0, 0, 255), fill_color:tuple=(0, 255, 0)):
        self.roi = self._roi_validation(roi)
        self.preprocess = Preprocessor(self.roi, pre_configs)
        self.generate = HoughLineGenerator(self.roi, gen_configs)
        self.studio = StudioManager(source, stroke_color, fill_color)
        
