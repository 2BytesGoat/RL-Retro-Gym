import cv2
import os

class CustomMonitor():
    def __init__(self, directory, size, fps=60, resume=False):
        self.directory = directory
        self.size = size
        self.fps = fps

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = None

        self.stem = 'custom-monitor'

        os.makedirs(self.directory, exist_ok=True)

        if not resume:
            self._clear_folder()

    def _clear_folder(self):
        for path in os.listdir(self.directory):
            if self.stem + "." in path:
                os.remove(os.path.join(self.directory, path))

    def start_new(self, name):
        if self.writer:
            self.writer.release()
        self.writer = cv2.VideoWriter(f'{self.directory}/{self.stem}.{name}.avi', self.fourcc, self.fps, self.size)

    def write(self, frame, to_bgr=True):
        if to_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)
        
    def release(self):
        self.writer.release()