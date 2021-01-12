"""
More info on data loaders: 
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import torch.utils.data
from pathlib import Path
from skimage import io

class SuperMarioKartDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=[]):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_file_paths = self._get_image_paths_from_dir(root_dir)

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        image_path = self.image_file_paths[idx]
        image_sample = io.imread(image_path)
        
        for transform in self.transforms:
            image_sample = transform(image_sample)

        return image_sample

    def _get_image_paths_from_dir(self, root_dir, suffix='png'):
        return Path(root_dir).glob(f'*.{suffix}')