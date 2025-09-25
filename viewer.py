import cv2
import numpy as np

class MediaViewer():

    _DEFAULT_CONFIG = {'maxWidth': 800}

    def __init__(self, sourceObject, processorObject, configs):

        if configs is None:
            configs = self._DEFAULT_CONFIG

        self.source = sourceObject
        self.processor = None if processorObject is None else processorObject
        self.writer = None
        self.fileOutName = None ### CONFIGURE OPTION TO ALLOW USERS TO PASS CUSTOM FILE-OUT NAME
        self.maxWidth = configs['maxWidth']
        self.paused = False
        self.exit = False
        self.currentFrame = 0

    def returnFrame(self):
        if self.source.sourceType == 'image':
            return True, self.source.image.copy()
        
        else:
            if self.source.cap is None:
                return False, None

            ret, frame = self.source.cap.read()

            if ret:
                return True, frame
            else:
                return False, None
    
    def videoStream(self, processor, compare, save):

        if save:
            self._writeMedia()

        print(f"Playing {self.source.name}. {self.source.name} contains {self.source.frameCount} frames.")
        self._printTransportControls()
        
        while True and not self.exit:
            ret, raw = self.returnFrame()
            if ret:
                if processor:
                    if self.processor is not None:
                        processed = self.processor.imageProcessor(raw)
                        if compare:
                            frame = self._renderDiptych([raw, processed])
                        else:
                            frame = self._renderFrame(processed)
                    else:
                        raise ValueError("Error: User must pass processor object to MediaVeiwer().")
                else:
                    frame = self._renderFrame(raw)
                
                if save:
                    self.writer.write(frame)

                cv2.imshow(self.source.name, frame)
                self._playbackControls()
            else:
                self._cleanUp()
                break
    
    def staticImage(self, media, write):

        if len(media) == 1:
            media = media[0]
        elif len(media) == 2:
            media = self._renderDiptych(media)
        elif len(media) == 4:
            media = self._renderMosaic(media)

        if write:
            self._writeMedia(media, self.fileOutName)

        cv2.imshow(self.source.name, media)
        cv2.waitKey(0)

        self._cleanUp()

    def _resizeFrame(self, img):
        if self.source.width > self.maxWidth:
            ratio = self.maxWidth / self.source.width
            w = self.maxWidth
            h = int(ratio * self.source.height)
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        else:
            return img

    def _channelChecker(self, img):
        if len(img.shape) < 3:
            return cv2.merge([img, img, img])
        else:
            return img

    def _renderFrame(self, img):
        img = self._resizeFrame(img)
        img = self._channelChecker(img)
        return img
        
    def _renderDiptych(self, frames):
        rendered = [self._renderFrame(frame) for frame in frames]

        diptych = np.hstack(rendered)
        h, w = diptych .shape[:2]
        cv2.line(diptych, (w // 2, 0), (w // 2, h), (0, 0, 0), 1, cv2.LINE_AA)
        return diptych
    
    def _renderMosaic(self, frames):
        top = self._renderDiptych(frames[:2])
        bottom = self._renderDiptych(frames[2:])

        mosaic = np.vstack([top, bottom])
        h, w = mosaic.shape[:2]
        cv2.line(mosaic, (0, h // 2), (w, h // 2), (0, 0, 0), 1, cv2.LINE_AA)
        return mosaic

    def _writeMedia(self):
        self.fileOutName = self.source.name + '-processed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        
        self.writer = cv2.VideoWriter(self.fileOutName, fourcc, self.source.fps, (self.source.width, self.source.height))

    def _printTransportControls(self):
        print("-------------------------------------------")
        print("\t\033[1;4mTransport Controls\033[0m\n")
        print("     \033[1mCommand      | Wait Key\033[0m")
        print("     --------------------------")
        print("     \033[3mQuit\033[0m         : 'q' or ESC")
        print("     \033[3mPause/Resume\033[0m : 'p' or SPACE")
        print("     \033[3mFast-Forward\033[0m : 'f'")
        print("     \033[3mRewind\033[0m       : 'r'")
        print("     \033[3mHelp\033[0m         : 'h'")
        print("------------------------------------------")
    
    def _playbackControls(self):
        while self.paused:
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('p'), ord(' ')]:
                print("Resuming video.")
                self.paused = False
            elif key in [ord('q'), ord('Q'), 27]:
                print("User entered quit. Exiting video player.")
                self.exit = True
                break
            elif key == ord('r'):
                print(f"Pressed rewind. Skipping to frame {0 if self.currentFrame - 50 <= 0 else self.currentFrame - 50}")
                self.currentFrame = max(0, self.currentFrame - 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrame)
            elif key == ord('f'):
                print(f"Pressed fast-forward. Skipping to frame {self.source.frameCount - 1 if self.currentFrame + 50 > self.source.frameCount - 1 else self.currentFrame + 50}")
                self.currentFrame = min(self.source.frameCount - 1, self.currentFrame + 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrame)
        
        if not self.paused:
            key = cv2.waitKey(self.source.fps) & 0xFF
            if key in [ord('p'), ord(' ')]:
                self.paused = True
                print(f"Pressed pause at frame {self.currentFrame}.")

            elif key in [ord('q'), ord('Q'), 27]:
                print("User entered quit. Exiting video player.")
                self.exit = True
            
            elif key == ord('r'):
                print(f"Pressed rewind. Skipping to frame {0 if self.currentFrame - 50 <= 0 else self.currentFrame - 50}")
                self.currentFrame = max(0, self.currentFrame - 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrame)

            elif key == ord('f'):
                print(f"Pressed fast-forward. Skipping to frame {self.source.frameCount - 1 if self.currentFrame + 50 > self.source.frameCount - 1 else self.currentFrame + 50}")
                self.currentFrame = min(self.source.frameCount - 1, self.currentFrame + 50)
                self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrame)
        
            elif key == ord('h'):
                print("Pressed help. Displaying 'Transport Controls'.")
                self._printTransportControls()
            
            self.currentFrame += 1

    def _cleanUp(self):
        if self.source.cap is not None:
            self.source.cap.release()
            self.source.cap = None
            print(f"Released {self.source.sourceType} capture object.")
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Released {self.source.sourceType} writer object.")

    def __del__(self):
        print("Executing clean-up")
        self._cleanUp()
        cv2.destroyAllWindows()
        print("Windows destroyed.")
