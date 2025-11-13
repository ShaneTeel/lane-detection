import cv2

from numpy.typing import NDArray
from typing import Literal, Union

from lane_detection.studio import StudioManager
from lane_detection.utils import ROISelector
from lane_detection.preprocessing import CannyPreprocessor, HoughPreprocessor
from lane_detection.detector import KalmanOLSLaneDetector, KalmanRANSACLaneDetector, OLSLaneDetector, RANSACLaneDetector

class LaneDetectionSystem():

    _BOLD = "\033[1m"
    _ITALICS = "\033[3m"
    _UNDERLINE = "\033[4m"
    _END = "\033[0m"

    def __init__(self, source:Union[str, int], roi:NDArray, preprocessor:Union[CannyPreprocessor, HoughPreprocessor], detector1:Union[KalmanOLSLaneDetector, KalmanRANSACLaneDetector, OLSLaneDetector, RANSACLaneDetector], detector2:Union[KalmanOLSLaneDetector, KalmanRANSACLaneDetector, OLSLaneDetector, RANSACLaneDetector]):        
        self.studio = StudioManager(source)
        self.fps = self.studio.get_fps()
        self.mask = ROISelector(roi)
        self.preprocessor = preprocessor
        self.detector1 = detector1
        self.detector2 = detector2
        self.name = f"{self.studio.get_name()} {detector1.name}"
        self.exit = False

    def run(self, view_style:Union[Literal["inset", "mosaic", "composite"], None]="composite", stroke:bool=False, fill:bool=True, save:bool=False):
        frame_names = self.configure_output(view_style, save)
        cv2.namedWindow(self.name)

        while True and not self.exit:
            ret, frame = self.studio.return_frame()
            if not ret:
                break
            else:
                masked = self.mask.inverse_mask(frame)
                thresh, edge_map, left, right = self.preprocessor.preprocess(masked, self.mask.x_mid)

                left = self.detector1.detect(left)
                right = self.detector2.detect(right)

                frame_lst = [frame, thresh, edge_map]
                lane_lines = [left, right]
                final = self.studio.gen_view(frame_lst, frame_names, lane_lines, view_style, stroke, fill)
                self.generate_output(view_style, save, final)
        report = self.generate_report()
        return report
    
    def configure_output(self, view_style:str=None, save:bool=False):
        if view_style is not None:      
            if self.studio.source_type() != "image":
                self.studio.print_menu()
            if save:
                self.studio.create_writer()
            return self.studio.get_frame_names(view_style.lower())

    def generate_output(self, view_style:str, save:bool, frame:NDArray): 
        if view_style is not None:
            cv2.imshow(self.name, frame)
            if self.studio.control_playback():
                self.exit = True
        if save:
            self.studio.write_frames(frame)
    
    def generate_report(self):
        metrics1 = self.detector1.return_metrics()
        metrics2 = self.detector2.return_metrics()
        report = f"\n{self._BOLD}{self._UNDERLINE}{self._ITALICS}{self.name} Report{self._END}\n\n"
        report += f"{self._BOLD}{self._ITALICS}Metrics      Left     Right{self._END}\n"
        for key in metrics1.keys():
            report += f"{key:>7}{metrics1[key]:>10.4f}{metrics2[key]:>10.4f}\n"
        return report
        