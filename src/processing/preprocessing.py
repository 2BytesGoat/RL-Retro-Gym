import cv2

def fetch_region_pipeline(
        grayscale=False, 
        cropper=False, cropper_roi=None, 
        resize=False, resize_shape=[25, 25]
    ):
    
    processes = []
    if grayscale:
        processes.append(('grayscale', ToGrayscale()))
    if cropper:
        processes.append(('cropper', Cropper(cropper_roi)))
    if resize:
        processes.append(('resizer', Resizer(resize_shape)))

    pipeline = Pipeline(processes)
    return pipeline

class Pipeline:
    def __init__(self, components):
        self.components = components

    def fit(self, X, y=None):
        for component_name, component in self.components:
            component.fit(X, y)
        return self

    def transform(self, X, y=None):
        new_X = X
        for component_name, component in self.components:
            new_X = component.transform(new_X)
        return new_X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

class Cropper:
    def __init__(self, roi):
        # assert len(roi) == 4 
        self.roi = roi

    def __crop_to_roi(self, image):
        return image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.__crop_to_roi(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

class ToGrayscale:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

class Resizer:
    def __init__(self, shape):
        self.shape = shape

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return cv2.resize(X, self.shape)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)