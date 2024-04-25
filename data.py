# random

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import nibabel as nib
import random 

import pickle
import util


class trainSet(Dataset):
    def __init__(self, data_root, args=None,
                random_crop=None, resize=None, augment_s=True, augment_t=True):
        self.args = args
        self.data_root = data_root
        self.folder_list = [(data_root + '/' + f) for f in os.listdir(data_root)]
        random.shuffle(self.folder_list)
        
        self.file_len = len(self.folder_list)

        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t
        

    def __getitem__(self, index):
        volumepath = self.folder_list[index]
        slice_list = [(volumepath+'/'+f) for f in os.listdir(volumepath)]
        slice_list.sort()
        
        def getAVol(slice_list):
            id = random.randint(0,len(slice_list) - self.args.hr_slice_patch)
            volume = []
            for i in range(self.args.hr_slice_patch):
                with open(slice_list[id+i], 'rb') as _f: img = pickle.load(_f)
                volume.append(img['image'])
                # img = np.load(slice_list[id+i])
                # volume.append(img)

            volume = np.array(volume,dtype=np.float32).transpose(1,2,0) 
            volume = util.normalize(volume) # [0-4095]->[0-1]
            if random.random() >= 0.5:
                volume = volume[:,::-1,:].copy()
            if random.random() >= 0.5:
                volume = volume[::-1,:,:].copy()

            volume = util.crop_center(volume,256,256) #[256,256,7]
            volume=torch.from_numpy(volume)
            return volume

        volume = []
        for i in range(self.args.one_batch_n_sample):
            volume.append(getAVol(slice_list))
        volume = torch.stack(volume,0)
        
        return volume

    def __len__(self):
        return self.file_len


class testSet(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.trainlist = [(data_root + '/' + f) for f in os.listdir(data_root)]

        self.file_len = len(self.trainlist)
         
    def __getitem__(self, index):
        volumepath = self.trainlist[index]
        with open(volumepath, 'rb') as _f: volumeIn = pickle.load(_f)
        volumeIn = volumeIn['image'] #[h,w,s] [0,4095]
        volumeIn = util.crop_center(volumeIn,256,256)
        volumeIn = util.normalize(volumeIn).astype(np.float32)
        volumeIn=torch.from_numpy(volumeIn) # w,h,s
        
        name = volumepath.split('/')[-1].split('.')[0]
        return name,volumeIn # [h,w,slice]

    def __len__(self):
        return self.file_len



