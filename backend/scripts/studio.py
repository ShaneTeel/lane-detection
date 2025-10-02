import cv2
import numpy as np

class Render():

    def __init__(self, source_object):

        self.source = source_object
        self.current_frame = 0

    def single_panel(self, processor = None):
        while True:
            ret, frame = self.source.return_frame()
            if ret:
                if processor is not None:
                    _, _, frame = processor.run(frame)
                
                self.source.
                    
            else:
                break  

    def create_mosaic(self, containers, processor):
        self._create_playback_options()
        while True and not self.exit:
            ret, raw = self.source.return_frame()
            if ret:
                thresh, edge_map, composite = processor.run(raw)
                frame = self._render_mosaic([raw, thresh, edge_map, composite])

                containers.image(frame, channels='BGR', use_container_width=True)
                self.current_frame += 1
            else:
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def create_finished(self, containers, processor):
        self._create_playback_options()
        while True and not self.exit:
            ret, raw = self.source.return_frame()
            if ret:
                _, _, composite = processor.run(raw)

                containers.image(composite, channels='BGR', use_container_width=True)
                self.current_frame += 1
            else:
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

class Render():
    def __init__(self):

        pass

    def single_frame(self, frame, processor = None):
        while True:
            ret, raw = self.source.return_frame()
            if ret:
                cv2.waitKey(self.source.fps)
                self.current_frame += 1
            else:
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
        pass

    def _render_diptych(self, frames):
        frames = [self._channel_checker(frame) for frame in frames]
        diptych = np.hstack(frames)
        h, w = diptych .shape[:2]
        cv2.line(diptych, (w // 2, 0), (w // 2, h), (0, 0, 0), 1, cv2.LINE_AA)
        return diptych
    
    def _render_mosaic(self, frames):
        top = self._render_diptych(frames[:2])
        bottom = self._render_diptych(frames[2:])

        mosaic = np.vstack([top, bottom])
        h, w = mosaic.shape[:2]
        cv2.line(mosaic, (0, h // 2), (w, h // 2), (0, 0, 0), 1, cv2.LINE_AA)
        return mosaic
    
    def _channel_checker(self, frame):
        if len(frame.shape) < 3:
            return cv2.merge([frame, frame, frame])
        else:
            return frame