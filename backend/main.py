from fastapi import FastAPI, HTTPException, Response, File, UploadFile, Depends
from typing import List
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import os
import asyncio
import uvicorn
from typing import List, Dict
from scripts.studio import Render, Read, Write
from scripts.CannyHoughP import CannyHoughP

app = FastAPI()

class AppState:

    def __init__(self):
        self.data = {}

    def add_item(self, key, value):
        self.data[key] = value

    def get_attr(self, key):
        return self.data.get(key)
    
state = AppState()

def get_state():
    return state

@app.get("/")
def read_root():
    return {"message": "CannyHoughP Lane-Detection API", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/initialize")
def create_source(file: UploadFile = File(...)):
    try:
        source = Read(file)
        state.add_item("source", source)
        render = Render()
        state.add_item("render", render)
        ret, frame = source.return_frame()
        if not ret:
            error = f"Error: Could not read frame from {source.name}"
            raise HTTPException(status_code=500, detail=error)
        else:
            ret, im = cv2.imencode(".jpeg", frame)
            return Response(im.tobytes(), media_type="image/jpeg")
        

    except Exception as e:
        print(f"Error occurred creating source object: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roi")
def define_roi(request: dict):
    points = request.get('points')
    top = min(*[y for _, y in [point for point in points]])
    bottom = max(*[y for _, y in [point for point in points]])
    mid_y = sum([top, bottom]) // 2

    left = min(*[x for x, _ in [point for point in points]])
    right = max(*[x for x, _ in [point for point in points]])
    mid_x = sum([left, right]) // 2
    
    roi = ['TL',
           'TR',
           'BR',
           'BL',
           'TL']
    
    for point in points:
        x, y = point
        if x < mid_x and y < mid_y:
            roi[0] = point
            roi[-1] = point
        elif x > mid_x and y < mid_y:
            roi[1] = point
        elif x > mid_x and y > mid_y:
            roi[2] = point
        elif x < mid_x and y > mid_y:
            roi[3] = point

    state.add_item("roi", roi)

    return {"poly": roi[:4]}

@app.post("/configure")
def configure_processor(request: dict):
    try:
        print("Trying")
        processor = CannyHoughP(request)
        state.add_item("processor", processor)
        
    except Exception as e:
        error = f"Error occured while processing frames: {str(e)}"
        raise HTTPException(status_code=500, detail=error)

async def render_frame(style: str, resource: AppState = state):
    source = resource.get_attr('source')
    processor = resource.get_attr('processor')
    render = resource.get_attr('render')

    source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_names = ["Threshold", "Edge Map", "Hough Lines", "Final Composite"]

    while True:
        ret1, raw = source.return_frame()
        if not ret1:
            source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        thresh, edge, hough, composite = processor.run(raw)
        if style == 'Step-by-Step':
            frame = render.render_mosaic([thresh, edge, hough, composite], frame_names)
        else:
            frame = composite
        
        ret2, buffer = cv2.imencode(".jpg", frame)
        if not ret2:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        await asyncio.sleep(0)

@app.get("/stream_video")
def stream_video(style: str):
    try:
        return StreamingResponse(
            content=render_frame(style),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
        
    except Exception as e:
        error = f"Error occured while processing frames: {str(e)}"
        raise HTTPException(status_code=500, detail=error)
    
@app.get("/process_frame")
def process_frame(style: str, idx: int):
    resource = get_state()
    source = resource.get_attr('source')
    processor = resource.get_attr('processor')
    render = resource.get_attr("render")
    source.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    ret, raw = source.cap.read()

    if not ret:
        raise ValueError(f"Error: could not encode frame: {frame}")
    frames = []
    frames.append(processor.run(raw))
    frame_names = ["Threshold", "Edge Map", "Hough Lines", "Final Composite"]
    if style == 'Step-by-Step':
        frame = render.render_mosaic(frames, frame_names)
    else:
        frame = frame[-1]
    
    ret2, buff = cv2.imencode(".png", frame)

    if not ret2:
        raise ValueError(f"Error: could not encode frame: {frame}")
    
    return Response(buff.tobytes(), media_type="image/png")


