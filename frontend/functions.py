import os
import cv2
import tempfile

def initialize_source(uploaded_file):
    video_bytes = uploaded_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        f.write(video_bytes)

    return cv2.VideoCapture(f.name)