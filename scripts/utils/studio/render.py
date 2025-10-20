import cv2
import numpy as np

class Render:

    def render_final_view(self, frames:list, names:list = None):
        frames = [self._channel_checker(frame) for frame in frames]
    
        if len(frames) % 2 != 0:
            blank = np.zeros_like(frames[0])
            frames.append(blank)
            names.append("Blank")

        if names is not None:
            for i in range(len(frames)):
                self._draw_banner_text(frames[i], names[i])

        top = self._render_diptych(frames[:2])
        bottom = self._render_diptych(frames[2:])

        mosaic = np.vstack([top, bottom])
        h, w = mosaic.shape[:2]
        cv2.line(mosaic, (0, h // 2), (w, h // 2), (0, 255, 255), 1, cv2.LINE_AA)
        return mosaic   

    def _render_diptych(self, frames):
        diptych = np.hstack(frames)
        h, w = diptych .shape[:2]
        cv2.line(diptych, (w // 2, 0), (w // 2, h), (0, 255, 255), 1, cv2.LINE_AA)
        return diptych

    def _channel_checker(self, frame):
        if len(frame.shape) < 3:
            return cv2.merge([frame, frame, frame])
        else:
            return frame
    
    def _draw_banner_text(self, frame, text):
        h, w = frame.shape[:2]
        banner_height = int(0.08 * h)
        cv2.rectangle(frame, (0, 0), (w, banner_height), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

        cv2.putText(frame, text, ((w // 2) - 75, 10 + (banner_height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)