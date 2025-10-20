from .studio import Reader, Writer, Illustrator, Render, Controller, Custodian

class StudioManager():
    
    def __init__(self, source, stroke_color:tuple = None, fill_color:tuple = None):

        self.source = Reader(source)
        self.render = Render()
        self.write = Writer(self.source)
        self.draw = Illustrator(stroke_color=stroke_color, fill_color=fill_color)
        self.playback = Controller(source)
        self.clean = Custodian(self.source, self.write)

