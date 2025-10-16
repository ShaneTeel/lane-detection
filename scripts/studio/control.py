import cv2

class Controller:

    def __init__(self, source):
        self.paused = False
        self.exit = False
        self.source = source
        self.current_frame = 0

    def _playbackControls(self):
        while self.paused:
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('p'), ord('P'), ord(' ')]:
                print("Resuming video.")
                self.paused = False
            elif key in [ord('q'), ord('Q'), 27]:
                print("Exiting video player.")
                self.exit = True
                break
            elif key == ord('r'):
                print(f"Skipping to frame {0 if self.current_frame - 50 <= 0 else self.current_frame - 50}")
                self.current_frame = max(0, self.current_frame - 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            elif key == ord('f'):
                print(f"Skipping to frame {self.source.frame_count - 1 if self.current_frame + 50 > self.source.frame_count - 1 else self.current_frame + 50}")
                self.current_frame = min(self.source.frame_count - 1, self.current_frame + 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        if not self.paused:
            key = cv2.waitKey(self.source.fps) & 0xFF
            if key in [ord('p'), ord(' ')]:
                self.paused = True
                print(f"Pausing at frame {self.current_frame}.")

            elif key in [ord('q'), ord('Q'), 27]:
                print("Exiting video player.")
                self.exit = True
            
            elif key == ord('r'):
                print(f"Skipping to frame {0 if self.current_frame - 50 <= 0 else self.current_frame - 50}")
                self.current_frame = max(0, self.current_frame - 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            elif key == ord('f'):
                print(f"Skipping to frame {self.source.frame_count - 1 if self.current_frame + 50 > self.source.frame_count - 1 else self.current_frame + 50}")
                self.current_frame = min(self.source.frame_count - 1, self.current_frame + 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            self.current_frame += 1