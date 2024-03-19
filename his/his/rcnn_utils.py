import torch
from pathlib import Path
import os
from torchvision import io
import pickle
import numpy as np
from torch import nn
import random
from torch.utils.data import Dataset, DataLoader, random_split
from collections import namedtuple
from torchvision import tv_tensors

#%%
class BaseRcnnDataset(Dataset):
    def __init__(self, 
                 images_dir: str,
                 annotations_dir: str,
                 transforms = None):
        """Base loader for hierarchical dataset

        Args:
            images_dir (str): path to source images folder
            annotations_dir (str): path to annotation tree folder            
        """
        super().__init__()
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.images = os.listdir(images_dir)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # read image
        image = io.read_image(str(Path(self.images_dir) / self.images[index])).float()
        image /= 255
        
        # read annotations
        with open(str(Path(self.annotations_dir) /
                  self.images[index].replace(".png", ".pkl")), "rb") as f:
            annotation = pickle.load(f)
            annotation = [ann.astype(np.float) for ann in annotation]
        
        if self.transforms:
            image = self.transforms(image)
            annotation = [self.transforms(tv_tensors.Mask(torch.tensor(ann))) for ann in annotation]    

        return image, annotation


#%%
Data = namedtuple("Data", ["train", "val", "test"])