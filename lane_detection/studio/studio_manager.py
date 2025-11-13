from typing import Union
from numpy.typing import NDArray
import cv2

from lane_detection.studio.read import Reader
from lane_detection.studio.write import Writer
from lane_detection.studio.illustrate import Illustrator
from lane_detection.studio.render import Render
from lane_detection.studio.control import Controller
from lane_detection.studio.custodian import Custodian

class StudioManager():
    
    def __init__(self, source:Union[str, int], stroke_color:tuple = (0, 0, 255), fill_color:tuple = (0, 255, 0)):

        self.source = Reader(source)
        self.render = Render()
        self.write = Writer(self.source)
        self.draw = Illustrator(stroke_color=stroke_color, fill_color=fill_color)
        self.playback = Controller(self.source)
        self.clean = Custodian(self.source, self.write)
        self.exit = False

    def gen_view(self, frame_lst:list, frame_names:list, lines:list, view_style:str, stroke:bool=False, fill:bool=True):
        composite = self.draw.gen_composite(frame_lst[0], lines, stroke, fill)
        if view_style == "composite":
            return composite
        elif view_style == "inset":
            frames = [self.draw._draw_banner_text(frame, name) for frame, name in zip(frame_lst, frame_names)]
            return self.render.render_inset(composite, frames)
        else:
            frame_lst.append(composite)
            frames = [self.draw._draw_banner_text(frame, name) for frame, name in zip(frame_lst, frame_names)]
            return self.render.render_mosaic(frames)
        
    def return_frame(self):
        if self.source.source_type == 'image':
            return True, self.source.image
        
        if self.source.cap is None:
            return False, None

        ret, frame = self.source.cap.read()
        
        if ret:
            return True, frame
        else:
            return False, None

    def get_frame_names(self, view_style:str=None):
        view_style_names = {
            "inset": ["Original", "Threshold", "Edge Map"],
            "mosaic": ["Original", "Threshold", "Edge Map", "Composite"],
            "composite": ["Composite"]
        }
        if view_style is None:
            return "Composite"
        
        try:
            names = view_style_names[view_style]
            return names
        except Exception as e:
            raise KeyError(f"ERROR: Invalid argument passed to 'view_style'. Must be one of {[key for key in view_style_names.keys()]}")
        
    def get_fps(self):
        return self.source.fps
    
    def create_writer(self):
        self.write._initialize_writer()

    def write_frames(self, frame:NDArray):
        if self.write.writer:
            self.write.save_object(frame)
        else:
            raise RuntimeError("ERROR: Never created writer object")
        
    def print_menu(self):
        self.playback.print_playback_menu()
    
    def source_type(self):
        return self.source.source_type
    
    def control_playback(self):
        return self.playback.playback_controls()
    
    def get_name(self):
        return self.source.name