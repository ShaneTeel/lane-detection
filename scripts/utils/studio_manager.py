from .studio import Reader, Writer, Illustrator, Render, Controller, Custodian

class StudioManager():
    
    def __init__(self, source, stroke_color:tuple = None, fill_color:tuple = None, alpha:float=0.8, beta:float=0.3):

        self.source = Reader(source)
        self.render = Render()
        self.write = Writer(self.source)
        self.draw = Illustrator(stroke_color=stroke_color, fill_color=fill_color, alpha=alpha, beta=beta)
        self.playback = Controller(self.source)
        self.clean = Custodian(self.source, self.write)

    def gen_hough_view(self, frame_lst:list, frame_names:list, lines:list, view_style:str, stroke:bool=False, fill:bool=True):
        composite = self.draw._gen_hough_composite(frame_lst[0], lines, stroke, fill)
        if view_style == "composite":
            return composite
        elif view_style == "inset":
            frames = [self.draw._draw_banner_text(frame, name) for frame, name in zip(frame_lst, frame_names)]
            return self.render.render_inset(composite, frames)
        else:
            frame_lst.append(composite)
            frames = [self.draw._draw_banner_text(frame, name) for frame, name in zip(frame_lst, frame_names)]
            return self.render.render_mosaic(frames)

    def gen_ransac_view(self, frame_lst:list, frame_names:list, lines:list, view_style:str, stroke:bool=False, fill:bool=True):
        composite = self.draw._gen_ransac_composite(frame_lst[0], lines, stroke, fill)
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