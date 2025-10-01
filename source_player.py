import cv2
import numpy as np
import streamlit as st

class VideoPlayer():

    def __init__(self, source_object, processor_object):

        self.source = source_object
        self.processor = None if processor_object is None else processor_object
        self.paused = False
        self.exit = False
        self.current_frame = 0
    
    def video_stream(self, container):
        self._create_playback_options()
        while True and not self.exit:
            ret, raw = self.source.return_frame()
            if ret:
                if self.processor is not None:
                    frame = self.processor.run(raw)
                else:
                    frame = raw

                container.image(frame, channels='BGR')
            else:
                break
    
    def _create_playback_options(self):

        while self.paused:
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('p'), ord(' ')]:
                st.write("Resuming video.")
                self.paused = False
            elif key in [ord('q'), ord('Q'), 27]:
                st.write("Exiting video player.")
                self.exit = True
                break
            elif key == ord('r'):
                st.write(f"Skipping to frame {0 if self.current_frame - 50 <= 0 else self.current_frame - 50}")
                self.current_frame = max(0, self.current_frame - 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            elif key == ord('f'):
                st.write(f"Skipping to frame {self.source.frame_count - 1 if self.current_frame + 50 > self.source.frame_count - 1 else self.current_frame + 50}")
                self.current_frame = min(self.source.frame_count - 1, self.current_frame + 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        if not self.paused:
            key = cv2.waitKey(self.source.fps) & 0xFF
            if key in [ord('p'), ord(' ')]:
                self.paused = True
                st.write(f"Pressed pause at frame {self.current_frame}.")

            elif key in [ord('q'), ord('Q'), 27]:
                st.write("Exiting video player.")
                self.exit = True
            
            elif key == ord('r'):
                st.write(f"Skipping to frame {0 if self.current_frame - 50 <= 0 else self.current_frame - 50}")
                self.current_frame = max(0, self.current_frame - 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            elif key == ord('f'):
                st.write(f"Skipping to frame {self.source.frame_count - 1 if self.current_frame + 50 > self.source.frame_count - 1 else self.current_frame + 50}")
                self.current_frame = min(self.source.frame_count - 1, self.current_frame + 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            else:
                self.current_frame += 1