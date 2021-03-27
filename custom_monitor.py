import cv2
import os
import torch

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

    def write(self, frame, state=None, enc_state=None, info=None, to_bgr=True):
        tmp_frame = frame.copy()
        # tmp_state = state
        # if state is not None and torch.is_tensor(state):
        #     tmp_state = tmp_state.detach().cpu().numpy()[0]
        # if state is not None:
        #     tmp_state = cv2.resize(tmp_state, frame.shape[:-1][::-1])
        #     tmp_frame = # make black and white
        #     tmp_frame = cv2.hconcat([tmp_frame, tmp_state])
        #     cv2.imwrite('test.png', tmp_frame)
        if to_bgr:
            tmp_frame = cv2.cvtColor(tmp_frame, cv2.COLOR_RGB2BGR)
        self.writer.write(tmp_frame)
        
    def release(self):
        self.writer.release()