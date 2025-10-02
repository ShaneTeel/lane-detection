import cv2
import streamlit as st
import requests
import time
from functions import initialize_source, clean_up

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
if 'container_lst' not in st.session_state:
    st.session_state['container_lst'] = []
if 'temp_file_path' not in st.session_state:
    st.session_state["temp_file_path"] = None

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
    cap = initialize_source(st.session_state['file'])
    st.session_state['cap'] = cap
    if not cap.isOpened():
        st.error(f"Failed to read uploaded file, {st.session_state['file'].name}")
        st.stop()
if st.session_state['file'] is not None:

    st.header("Step 1: Configure Model")
    cols1 = st.columns(2, border=True)

    # ROI Selection
    with cols1[0]:
        st.subheader("Select a Region of Interest* (ROI)")
        st.markdown("**Move cursor over image and right-click at four different points**")
        st.write("**The ROI is the area of the frame that the Lane Detection model is run against.*")

        # Create ROI Window Session State
        if st.session_state['roi_window'] is None:
            st.session_state['roi_window'] = st.empty()
            st.session_state['container_lst'].append(st.session_state['roi_window'])
            ret, st.session_state['roi_frame'] = st.session_state['cap'].read()
        st.session_state['roi_window'].image(st.session_state['roi_frame'], channels='BGR', use_container_width=True)  
            
    with cols1[1]:
        st.subheader("Set Parameters")

        # Thresholding Input (cv2.inRange())
        st.markdown("**Thresholding**")
        cols1A = st.columns(2)
        with cols1A[0]:
            lower_bounds = st.number_input("Lower Bounds (Inclusive)", min_value=0, max_value=254, value=150)
        with cols1A[1]:
            upper_bounds = st.number_input("Lower Bounds (Inclusive)", min_value=0 if lower_bounds is None else lower_bounds + 1, max_value=255, value=255 if lower_bounds is None else lower_bounds + 1) 

        # Canny Input (cv2.Canny())
        st.markdown("**Canny Edge Detection**")
        cols1B = st.columns(3)
        with cols1B[0]:
            canny_low = st.number_input("Hysteresis Lower Bounds (Inclusive)", min_value=0, max_value=300, value=50)
        with cols1B[1]:
            canny_high = st.number_input("Hysteresis Upper Bounds (Inclusive)", min_value=0 if canny_low is None else canny_low + 1, max_value=300, value=100 if canny_low is None else canny_low + 1)
        with cols1B[2]:
            blur_first = st.selectbox("Gaussian Blur", ["Before Canny", "After Canny"])
        
        # Hough Input (cv2.HoughLinesP())
        st.markdown("**Probabilistic Hough Line Transform**")
        cols1C = st.columns(5)
        with cols1C[0]:
            
        st.markdown("**Composite Styling***")

        st.write("*Style options do not affect the output of the algorith, only the style in which the lanes are drawn onto the image.")

    view_options = ['Step-by-Step', 'Fully Processed']
    st.divider()
    st.header("Step 2: Inspect & Evaluate")
    cols2 = st.columns(2, border=True)
    with cols2[0]:
        st.subheader("Visual Inspection")
        view_selection = st.segmented_control("Render Options", view_options)
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
