"""
More info on data loaders: 
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import torch.utils.data
from pathlib import Path
from skimage import io

class SuperMarioKartDataset(torch.utils.data.Dataset):
    def __init__(self, meta_txt, transforms=[]):
        self.transforms = transforms
        self.image_file_paths = self._meta_to_list(meta_txt)
        self.image_shape = self.__getitem__(0).shape # assume all frames have the same shape

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        image_path = self.image_file_paths[idx]
        image_sample = io.imread(image_path)
        
        for transform in self.transforms:
            image_sample = transform(image_sample)

        return image_sample

    def _meta_to_list(self, meta_txt):
        """
        Takes a text file of filenames and makes a list of filenames
        """
        with open(meta_txt, encoding="utf-8") as f:
            files = f.readlines()

        files = [f.rstrip() for f in files]
        return files