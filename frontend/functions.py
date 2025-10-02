import os
import cv2
import tempfile

def initialize_source(source):
    video_bytes = source.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        f.write(video_bytes)

    return cv2.VideoCapture(f.name)

def clean_up(cap, container_lst):
    cap.release()
    for container in container_lst:
        container.empty()
