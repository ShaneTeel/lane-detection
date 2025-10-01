import streamlit as st


def _create_playback_options(self):

        playback_speed = {
            "Subtract": ":material/remove:",
            "Reset": ":material/restart_alt:",
            "Add": ":material/add:"
        }

        st.segmented_control(
            'Playback Speed',
            options=playback_speed.keys(),
            format_func=lambda option: playback_speed[option],
            selection_mode='single'
        )

        playback_control

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