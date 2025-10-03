import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import requests
import time
from functions import initialize_source

# Page Layout
st.set_page_config(layout='wide')

if 'reset' not in st.session_state:
    st.session_state['reset'] = True
if 'file' not in st.session_state:
    st.session_state['file'] = None
if 'cap' not in st.session_state:
    st.session_state['cap'] = None
if 'roi_window' not in st.session_state:
    st.session_state['roi_window'] = None
if 'roi_frame' not in st.session_state:
    st.session_state['roi_frame'] = None
if 'roi_points' not in st.session_state:
    st.session_state['roi_points'] = []
if 'roi_canvas' not in st.session_state:
    st.session_state['roi_canvas'] = None
if 'container_lst' not in st.session_state:
    st.session_state['container_lst'] = []

# Set title
st.title("Traditional Lane Detection Walk-Through")

# Create file upload button
st.header("Source Management")
cols0 = st.columns(2)
with cols0[0]:
    st.subheader("Video Upload")
    uploaded_file = st.file_uploader("Choose a source file:", type=['mp4', 'mov', 'mkv', 'avi'], key=f"file_uploader_{st.session_state['reset']}")
    
with cols0[1]:
    st.subheader("Video Release")
    st.write("Select after ")
    release = st.button("Release", help="Release all capture objects and empty all containers", type='primary')

# Verify and Read Uploaded File; Initialize Capture Object 
if uploaded_file is not None and uploaded_file != st.session_state['file']:
    st.spinner("Initializing capture object...")
    st.session_state['file'] = uploaded_file
    video_bytes = uploaded_file.read()
    cap = initialize_source(video_bytes)
    st.session_state['cap'] = cap
    if not cap.isOpened():
        st.error(f"Failed to read uploaded file, {st.session_state['file'].name}")
        st.stop()
if st.session_state['file'] is not None:

    st.header("Step 1: Configure Model")
    cols1 = st.columns(2)

    # ROI Selection
    with cols1[0]:
        st.subheader("Select a Region of Interest* (ROI)")
        st.markdown("**Move cursor over image and right-click four different points on the image.**")
        st.write("**The ROI is the area of the frame that the Lane Detection model is run against.*")
        # Create ROI Session State
        if st.session_state['roi_frame'] is None:
            ret, roi_frame = st.session_state['cap'].read()
            roi_frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            st.session_state['roi_frame'] = Image.fromarray(roi_frame_rgb)
        if st.session_state['roi_canvas'] is None:
            st.session_state['roi_canvas'] = st_canvas(
                fill_color = "rgba(0, 0, 0, 0)",
                stroke_width = 2,
                stroke_color = "rgba(255, 0, 0, 1)",
                background_image = st.session_state['roi_frame'],
                width = st.session_state['roi_frame'].width,
                height = st.session_state['roi_frame'].width,
                drawing_mode = "polygon",
                key="canvas"
            )
        if st.session_state['roi_window'] is None:
            st.session_state['roi_window'] = st.empty()
            st.session_state['container_lst'].append(st.session_state['roi_window'])
        st.session_state['roi_window'].image(st.session_state['roi_canvas'].image_data)
            
    with cols1[1]:
        st.subheader("Set Parameters")

        cols1A = st.columns(2)
        
        # ROI Selection
        with cols1A[0]:
            st.markdown("**Selected ROI**")

        # Thresholding Input (cv2.inRange())
        with cols1A[1]:
            st.markdown("**Thresholding**")
            lower_bounds, upper_bounds = st.slider("Lower / Upper (Inclusive)", min_value=0, max_value=255, value=(150, 255))

        # Canny Input (cv2.Canny())
        st.markdown("**Canny Edge Detection**")
        cols1B = st.columns(2)
        with cols1B[0]:
            blur_first = st.selectbox("Gaussian Blur", ["Before Canny", "After Canny", "No Blur"], help="Whether to perform Gaussian Blur before edge detection, after edge detection, or not at all.")
        with cols1B[1]:
            canny_low, canny_high = st.slider("Weak / Sure Edge (Inclusive)", min_value=0, max_value=300, value=(50, 150))

        # Hough Input (cv2.HoughLinesP())
        st.markdown("**Probabilistic Hough Line Transform**")
        cols1C = st.columns(5)
        # with cols1C[0]:
            # rho = st. 
        st.markdown("**Composite Styling***")

        st.write("*Style options do not affect the algorithm.")

    view_options = ['Step-by-Step', 'Fully Processed']
    st.divider()
    st.header("Step 2: Inspect & Evaluate")
    cols2 = st.columns(2)
    with cols2[0]:
        st.subheader("Visual Inspection")
        view_selection = st.selectbox("Render Options", view_options)
        window = st.empty() # Create a placeholder to display frames
        if view_selection == view_options[0]:
            
            pass
        if view_selection == view_options[1]:
            
            pass
    
    with cols2[1]:
        st.subheader("Evaluation Report")

    if release:
        st.session_state['reset'] = not st.session_state['reset']
        st.session_state['file'] = None
        st.session_state['cap'].release()
        for container in st.session_state['container_lst']:
            container.empty()
        st.rerun()
