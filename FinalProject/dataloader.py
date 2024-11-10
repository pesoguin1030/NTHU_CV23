import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import glob
import pandas as pd

    
    
def get_dataloader(data_dir, batch_size=16, num_workers=8, mode='train'):
    assert (mode=='train' or mode=='test'), 'data loader\'s mode is only \'train\' or \'test\' can choose'
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Resize((224,224), antialias=True),
                    transforms.RandomHorizontalFlip(p=0.3),
                    # transforms.RandomRotation(degrees=(0, 180)),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Resize((224,224), antialias=True),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])            
                
    if mode == 'train':            
        train_dataset = birds_datasets(data_dir, transform, mode='train')  
        val_dataset = birds_datasets(data_dir, transform, mode='valid')
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
        return train_dataloader, val_dataloader
        
        
    else: # mode == 'test':                           
        test_dataset = birds_datasets(data_dir, transform_test, mode)
        val_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        return val_dataloader
        
        
def dataset_path_loading(data_dir, mode):
    if mode == 'test':
        img_path_list = glob.glob(os.path.join(data_dir, 'test', '*.jpg'))
        return img_path_list
    else:
        # read img_path_list from birds.csv 
        csv_path = os.path.join(data_dir, 'birds.csv')
        csv_df = pd.read_csv(csv_path)
        csv_df = csv_df[csv_df['data set']==mode]
        csv_df = csv_df.reset_index(drop=True)
        # print(csv_df)
        return csv_df

        
class birds_datasets(Dataset):      
    def __init__(self, data_dir, transform, mode):
        super().__init__()
        print('='*20)
        print("data loading...")
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir
        if mode =='test':
            self.dataset_csv_df = []
            self.img_path_list = dataset_path_loading(data_dir, mode)
        else:  #'train' or 'valid'
            self.dataset_csv_df = dataset_path_loading(data_dir, mode)
            self.img_path_list  = []


    def __len__(self,):
        if self.mode == 'test':
            return len(self.img_path_list)            
        else: # 'train' or 'test'
            return len(self.dataset_csv_df)
    
    def __getitem__(self, idx):
        if self.mode == 'test':
            img_path = self.img_path_list[idx]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, img_path.split('/')[-1]
            
        else: #'train' or 'valid':
            img_fname = self.dataset_csv_df['filepaths'].iloc[idx]
            label = int(self.dataset_csv_df['class id'].iloc[idx])
            img_path = os.path.join(self.data_dir, img_fname)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label
            