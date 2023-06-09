from src.base import BaseDataSet, BaseDataLoader
# from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json

class ImageDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.data_len = kwargs['data_len']
        # self.palette = pallete.get_voc_pallete(self.num_classes)
        super(ImageDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ["val", "test", "train"]:
            file_list = os.path.join(self.root, 'list', f"{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")
        img_name_list = np.loadtxt(file_list, dtype=str)
        if img_name_list.ndim == 2:
            return img_name_list[:, 0]
        self.dataset_len = len(img_name_list)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        self.files = img_name_list[:self.data_len]

    def _load_data(self, index):
        image_A_path    = os.path.join(self.root, 'A', self.files[index%self.data_len])
        image_B_path    = os.path.join(self.root, 'B', self.files[index%self.data_len])
        image_A         = np.asarray(Image.open(image_A_path), dtype=np.float32)
        image_B         = np.asarray(Image.open(image_B_path), dtype=np.float32)
        image_id        = self.files[index%self.data_len].split("/")[-1].split(".")[0]
        label_path  = os.path.join(self.root, 'label', self.files[index%self.data_len])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image_A, image_B, label, image_id
    
    def __len__(self):
        return self.data_len

class CDDataset(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN       = [0.485, 0.456, 0.406]
        self.STD        = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean']  = self.MEAN
        kwargs['std']   = self.STD
        
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')
        
        self.dataset = ImageDataset(**kwargs)

        super(CDDataset, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)
