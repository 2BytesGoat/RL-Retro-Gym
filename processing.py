import cv2
from functools import partial

def get_transforms(roi=None, grayscale=False):
    transofrms = []
    if roi is not None:
        transofrms.append(frame_to_roi_partial(roi))
    if grayscale:
        transofrms.append(frame_to_grayscale)
    return transofrms

def apply_transforms(frame, transforms):
    tmp_frame = frame.copy()
    for trans in transforms:
        tmp_frame = trans(tmp_frame)
    return tmp_frame

def frame_to_roi_partial(roi):
    return partial(frame_to_roi, roi)

def frame_to_roi(roi=None, frame=None):
    assert roi is not None, 'Region of Interest can not be None'
    return frame[roi[0]:roi[1], roi[2]:roi[3]]

def frame_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)