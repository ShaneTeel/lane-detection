import cv2
import numpy as np
import os
import io
import base64
import shutil
import tempfile

class Render:
    def render_mosaic(self, frames, names):
        frames = [self._channel_checker(frame) for frame in frames]
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


class Read():

    def __init__(self, source):
        self.source = source
        self.name = None
        self.temp_file_path = None
        self.ext = None
        self.cap = None
        self.width = None
        self.height = None
        self.fps = None
        self.frame_count = None

        self._initialize_source()

    def return_frame(self):
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            return True, frame
        else:
            return False, None

    def _initialize_source(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            try:
                shutil.copyfileobj(self.source.file, temp_file)
                self.temp_file_path = temp_file.name
            finally:
                self.source.file.close()
        self.cap = cv2.VideoCapture(self.temp_file_path)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Failed to open {self.name} video.")

        else:       
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.name, self.ext = os.path.splitext(os.path.basename(self.source.filename))

    # def _clean_up(self):
    #     if self.cap is not None:
    #         self.cap.release()
    #         self.cap = None

    # def __del__(self):
    #     self._clean_up()
    #     os.unlink(self.temp_file_path)

class Write():
    def __init__(self, file_out_name, ext, width, heigth, fps):
        self.file_out_name = file_out_name
        self.ext = ext
        self.width = width
        self.height = heigth
        self.fps = fps
        self.writer = None
        
        self._initialize_writer()
    
    # Create file download button
    def get_download_link(file, file_name, text):
        buffered = io.BytesIO()
        file.save(buffered, format='mp4')
        file_str = base64.b64encode(buffered.getvalue()).decode()
        href = f"<a href='data:file/txt;base64,{file_str}' download='{file_name}'>{text}</a>"
        return href

    def _initialize_writer(self):
        self.file_out_name += self.ext
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.file_out_name, fourcc, self.fps, (self.width, self.height))
    
    def _clean_up(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def __del__(self):
        self._clean_up()