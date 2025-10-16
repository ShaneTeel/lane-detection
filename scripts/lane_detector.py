import cv2
import numpy as np
import numpy.typing as npt
from studio import Reader, Writer, Render, Illustrator, Controller, Custodian
from line_generator import CannyRANSAC, CannyHoughP

class LaneDetector():

    def __init__(self, source, processor_type:str = ['CannyHoughP', 'CannyRANSAC'], roi:npt.NDArray = None, configs:dict = None, stroke_color:tuple = (0, 0, 255), fill_color:tuple=(0, 255, 0)):
        
        self.source = Reader(source)
        self.writer = Writer(self.source)
        self.processor = self._initialize_processor(processor_type, roi, configs)
        self.draw = Illustrator(stroke_color, fill_color)
        self.render = Render()
        self.controller = Controller(self.source)
        self.custodian = Custodian(self.source, self.writer)

    def detect(self, names:list = None):
        if names is None:
            names = ["Raw", "Thresh", "Edge", "Composite"]

        while True:
            ret, frame = self.source.return_frame()
            if not ret:
                break
            else:
                thresh, edge, lines = self.processor.run(frame)
                composite = self.draw.draw_curved_stroke_fill(frame, lines, stroke=False)
                final = self.render.render_final_view([frame, thresh, edge, composite], names)

                cv2.imshow("test", final)
                key = cv2.waitKey(1)
                if key == 27 or key == 32:
                    break
        
    def _initialize_processor(self, processor_type, roi, configs:dict):
        if not isinstance(processor_type, str):
            raise TypeError("ERROR: Argument passed to `processor_type` must be of str dtype.")
        
        processor_type = processor_type.lower()

        if processor_type not in ["cannyhoughp", "cannyransac"]:
            raise ValueError("ERROR: Argument passed to `processor_type` not a valid processor type.")
        
        return CannyHoughP(roi, configs) if processor_type == "cannyhoughp" else CannyRANSAC(roi, configs)


if __name__ == "__main__":
    source = "media/lane1-straight.mp4"
    # source = "media/test_img1.jpg"

    processor_type = "CannyRANSAC"
    detector = LaneDetector(source, processor_type)

    detector.detect()