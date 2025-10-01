import cv2
import os
import io
import base64
import time
import streamlit as st

@st.cache_resource
class SourceReader():

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
        self.writer = None
        self.message = st.empty()

        self._initialize_source()

    def return_frame(self):
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            return True, frame
        else:
            return False, None

    # Create file download button
    def get_download_link(file, file_name, text):
        buffered = io.BytesIO()
        file.save(buffered, format='mp4')
        file_str = base64.b64encode(buffered.getvalue()).decode()
        href = f"<a href='data:file/txt;base64,{file_str}' download='{file_name}'>{text}</a>"
        return href

    def _initialize_source(self):
        video_bytes = self.source.read()
        self.temp_file_path = 'temp_video.mp4'
        with open(os.path.join(self.temp_file_path), 'wb') as f:
            f.write(video_bytes)

        self.cap = cv2.VideoCapture(self.temp_file_path)

        if not self.cap.isOpened():
            self.message.error(f"Error: Failed to open {self.source.name} video.")
            st.stop()
        else:       
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.name, self.ext = os.path.splitext(os.path.basename(self.source.name))

            self.message.success(f"Successfully opened video {self.name} ({self.width}w x {self.height}h, {self.fps} FPS, {self.frame_count} Frames).")
            time.sleep(5.0)
            self.message.empty()

    def _initialize_writer(self, file_out_name):
        file_out_name += '-processed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(file_out_name, fourcc, self.fps, (self.width, self.height))
    
    def _clean_up(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print(f"Released {self.name} capture object.")
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Released {self.name} writer object.")
    
    def __del__(self):
        self._clean_up()
        os.remove(self.temp_file_path)