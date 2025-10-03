import os
import cv2
import tempfile

def initialize_source(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        f.write(video_bytes)

    return cv2.VideoCapture(f.name)

