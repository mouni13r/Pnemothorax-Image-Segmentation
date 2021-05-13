# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:49:40 2020

"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torchvision.transforms as trans
import torch.nn.functional as F

class lungData(Dataset):
    
    def __init__(self,args,im_list,masks_list, labels = [],transform = None,im_type = 'RGB'):
        self.im_list = im_list
        self.masks_list = masks_list
        self.labels = np.array(labels)
        
        self.num_classes = args.num_classes
        self.one_hot = args.one_hot
        self.transform = transform
        
        assert im_type in ['RGB','L']
        self.im_type = im_type
        
    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self,idx):
        im = Image.open(self.im_list[idx]).convert(self.im_type)
        mask = Image.open(self.masks_list[idx])

        if(self.transform):
            im = self.transform(im)
            mask = self.transform(mask)

        if(self.labels.any()):
            label= np.array([self.labels[idx]])
            label = torch.from_numpy(label)
            if(self.one_hot):
                label = F.one_hot(label,num_classes = self.num_classes)
            return im,mask,label
        else:
            return im,mask
    
def create_dataloaders(args):
    
    transforms = trans.Compose([
        trans.Resize(args.im_dim),
        trans.ToTensor(),
        trans.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    
    data = pd.read_csv(args.train_csv)
    
    train, val = train_test_split(data, test_size = args.val_split)
    
    im_train = [args.im_dir + im for im in train['im_path']]
    mask_train = [args.mask_dir + im for im in train['im_path']]

    im_val = [args.im_dir + im for im in val['im_path']]
    mask_val = [args.mask_dir + im for im in val['im_path']]

    im_save = [args.im_dir + im for im in data['im_path']]
    mask_save = [args.mask_dir + im for im in data['im_path']]

    train_labels = train['label']
    val_labels = val['label']
    
    train_data = lungData(args, im_train, mask_train, labels = train_labels, transform = transforms, im_type='RGB')
    val_data = lungData(args, im_val, mask_val, labels = val_labels, transform = transforms, im_type='RGB')
    save_data = lungData(args, im_save, mask_save, transform = transforms, im_type='RGB') 

    train_loader = DataLoader(train_data, batch_size = args.batch_sz, shuffle = True, num_workers = args.num_workers, drop_last = args.drop_last)
    val_loader = DataLoader(val_data, batch_size = args.batch_sz, shuffle = False, num_workers = args.num_workers, drop_last = args.drop_last)
    save_loader = DataLoader(save_data, batch_size = args.batch_sz, shuffle = False, num_workers = args.num_workers)

    return {'train': train_loader, 'val': val_loader, 'save':save_loader}

        
