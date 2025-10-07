from fastapi import FastAPI, HTTPException, Response, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import os
import asyncio
import uvicorn
from typing import List, Dict
from scripts.studio import Render, Read, Write
from scripts.CannyHoughP import CannyHoughP

app = FastAPI()

class ProcessorConfigs(BaseModel):
    '''Input for processor'''
    configs: dict

class StreamConfigs(BaseModel):
    '''Input for processor'''
    configs: dict

app_state = {
    "source": None,
    "processor": None,
    "lock": asyncio.Lock()
}

def get_source(key):
    return app_state.get(key)

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
        app_state['source'] = source
        ret, frame = source.return_frame()
        if not ret:
            error = f"Error: Could not read frame from {source.name}"
            raise HTTPException(status_code=500, detail=error)
        else:
            ret, im = cv2.imencode(".png", frame)
            return Response(im.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error occurred creating source object: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/configure")
def configure_processor(request: ProcessorConfigs):
    try:
        processor = CannyHoughP(request.configs)
        app_state['processor'] = processor
        
    except Exception as e:
        error = f"Error occured while processing frames: {str(e)}"
        raise HTTPException(status_code=500, detail=error)

async def render_frame(style: str):
    source = app_state.get('source')
    processor = app_state.get('processor')
    render = Render()
    source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret1, raw = source.return_frame()
        if not ret1:
            break

        thresh, edge, composite = processor.run(raw)
        if style == 'Step-by-Step':
            frame = render._render_mosaic([raw, thresh, edge, composite])
        else:
            frame = composite

        ret2, buffer = cv2.imencode(".png", frame)
        if not ret2:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        await asyncio.sleep(0)  # hand control back to the event loop

@app.post("/stream_video")
def stream_video(request: StreamConfigs):
    try:
        style = request.configs.get("style")
        return StreamingResponse(
            content=render_frame(style),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
        
    except Exception as e:
        error = f"Error occured while processing frames: {str(e)}"
        raise HTTPException(status_code=500, detail=error)