import cv2
import streamlit as st
import streamlit_image_coordinates as img_xy
from PIL import Image, ImageDraw
import requests
from functions import initialize_source

# Page Layout
st.set_page_config(layout='wide')

# Destructor Attributes
if 'reset' not in st.session_state:
    st.session_state['reset'] = True
if 'container_lst' not in st.session_state:
    st.session_state['container_lst'] = []

# Source Attributes
if 'file' not in st.session_state:
    st.session_state['file'] = None
if 'cap' not in st.session_state:
    st.session_state['cap'] = None

# ROI Attributes
if 'roi_window' not in st.session_state:
    st.session_state['roi_window'] = None
if 'roi_frame' not in st.session_state:
    st.session_state['roi_frame'] = None
if 'roi_points' not in st.session_state:
    st.session_state['roi_points'] = []
if 'roi_canvas' not in st.session_state:
    st.session_state['roi_canvas'] = None
if 'pil' not in st.session_state:
    st.session_state['pil'] = None

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
    st.write("Select Before Exiting")
    release = st.button("Release", help="Release all capture objects and empty all containers", type='primary')

st.divider()

# Verify and Read Uploaded File; Initialize Capture Object 
if uploaded_file is not None and uploaded_file != st.session_state['file']:
    st.spinner("Initializing capture object...")
    st.session_state['file'] = uploaded_file
    cap = initialize_source(uploaded_file)
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
        st.markdown("**Move cursor over image and right-click at four different points on the image.**")
        st.write("**The ROI is the area of the frame that the Lane Detection model is run against.*")
        # Create ROI Window
        if st.session_state['roi_frame'] is None:
            ret, roi_frame = st.session_state['cap'].read()
            if ret:
                roi_frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(roi_frame_rgb)
                st.session_state['roi_frame'] = roi_frame_rgb
        if st.session_state['roi_window'] is None:
            st.session_state['roi_window'] = st.empty()
            st.session_state['container_lst'].append(st.session_state['roi_window'])
        st.session_state['roi_window'].image(st.session_state['roi_frame'], channels='RGB')
        
        def add_point():
            raw_value = st.session_state['pil']
            value = raw_value['x'], raw_value['y']
            st.session_state['roi_points'].append(value)
        # for point in st.session_state['roi_points']:
        #     coords = 

    with cols1[1]:
        st.subheader("Set Parameters")

        # ROI Selection
        cols1A = st.columns(2)
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
        with cols1C[0]:
            w, h, = st.session_state['roi_frame'].shape[:2]
            diag = (w**2 + h**2)**0.5
            area = w * h
            rho = st.number_input("Rho (ρ)", min_value=0.1, max_value=diag, value=1.0)
        with cols1C[1]:
            theta = st.number_input("Theta (θ)", min_value=0, max_value=180, value=180, help="Value will be divided by π once passed to model.")
        with cols1C[2]:
            min_votes = st.number_input("Threshold", min_value=1, max_value=area, value=50)
        with cols1C[3]:
            min_line_length = st.number_input("Min. Line Length", min_value=1, max_value=int(diag), value=10)
        with cols1C[4]:
            max_line_gap = st.number_input("Max Line Gap", min_value=0, max_value=int(diag), value=20)
        
        st.markdown("**Composite Styling***")
        cols1D = st.columns(4)
        with cols1D[0]:
            stroke_bool = st.checkbox("Draw Lane Lines (Stroke)")
        with cols1D[1]:
            if stroke_bool:
                stroke_color = st.color_picker("Stroke Color", value="#FF0000")
        with cols1D[2]:
            fill_bool = st.checkbox("Draw Lane Area (Fill)")
        with cols1D[3]:
            if fill_bool:
                fill_color = st.color_picker("Fill Color", value="#00FF00")
        cols1E = st.columns(3)
        with cols1E[0]:
            st.write("*Style options do not affect the algorithm.")
        with cols1E[2]:
            submit = st.button("Submit", type='secondary')

    processor_configs = {
        "in_range": {
            "lower_bounds": lower_bounds, 
            "upper_bounds": upper_bounds
        },
        "canny": {
            "canny_low": canny_low, 
            "canny_high": canny_high, 
            "blur_first": blur_first
        },
        "hough": {
            "rho": rho, 
            "theta": theta, 
            "thresh": min_votes, 
            "min_length": min_line_length, 
            "max_gap": max_line_gap
        },
        "composite": {
            "stroke": stroke_bool, 
            "stroke_color": stroke_color,
            "fill": fill_bool,
            "fill_color": fill_color
        }
    }


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