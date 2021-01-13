import cv2
from functools import partial
from torchvision import transforms

def get_transforms(roi=None, grayscale=False, to_tensor=True):
    transofrms = []
    if roi is not None:
        transofrms.append(frame_to_roi(roi))
    if grayscale:
        transofrms.append(frame_to_grayscale)
    if to_tensor:
        transofrms.append(transforms.ToTensor())
    return transofrms

def frame_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def frame_to_roi(roi):
    return partial(_frame_to_roi, roi)

def _frame_to_roi(roi=None, frame=None):
    assert roi is not None, 'Region of Interest can not be None'
    return frame[roi[0]:roi[1], roi[2]:roi[3]]