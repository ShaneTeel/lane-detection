import cv2
import streamlit as st
from source_reader import SourceReader
from source_player import VideoPlayer

# Page Layout
st.set_page_config(layout='wide')

# Set title
st.title("Traditional Lane Detection")

# Create file upload button    
uploaded_file = st.file_uploader("Choose a source file:", type=['mp4', 'mov', 'mkv', 'avi'])

if uploaded_file is not None:
    source = SourceReader(uploaded_file)

    player = VideoPlayer(source, None)
else:
    st.write("Please upload a video file to proceed.")
    st.stop()


original = st.empty()
player.video_stream(original)

source._clean_up()
original.empty()