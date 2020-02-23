# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:54:20 2019

@author: yuxuan
"""
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import h5py


class COCO_Dataset(data.Dataset):
    def __init__(self, img_dir, coco_io, transform):
        self.root = img_dir
        self.coco = coco_io
        self.transform = transform
        
    def __len__(self):
        return len(self.coco)
    
    def __getitem__(self, index):
        if self.coco[index]['filename'][5] == 't':
            img_name = self.root + 'train2014/' + self.coco[index]['filename'][:-4] + '.jpg'
        else:
            img_name = self.root + 'val2014/' + self.coco[index]['filename'][:-4] + '.jpg'

        # if self.coco[index]['filename'][5] == 't':
        #     img_name = self.root + 'train/' + self.coco[index]['filename'][:-4] + '.jpg'
        # else:
        #     img_name = self.root + 'val/' + self.coco[index]['filename'][:-4] + '.jpg'
        image = Image.open(img_name)
        image = image.convert("RGB")
        image = self.transform(image)
        
        label = torch.LongTensor(self.coco[index]['outputs'])
        mask = torch.FloatTensor(self.coco[index]['mask'])
        gts_index = self.coco[index]['filename']

        sample = {'image': image, 'label': label, 'mask': mask, 'gts_index': gts_index}
        
        return sample
