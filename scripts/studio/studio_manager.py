from .read import Reader
from .write import Writer
from .illustrate import Illustrator
from .render import Render
from .control import Controller
from .custodian import Custodian

class StudioManager():
    
    def __init__(self, source, stroke_color:tuple = None, fill_color:tuple = None):

        self.source = Reader(source)
        self.render = Render()
        self.write = Writer(self.source)
        self.draw = Illustrator(stroke_color=stroke_color, fill_color=fill_color)
        self.playback = Controller(self.source)
        self.clean = Custodian(self.source, self.write)

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