import os
import pandas as pd
import torch 
from torch.utils.data import Dataset
import cv2 as cv

class CustomImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, col_images: str, col_annotations:str, device: str) -> None:
        self.images = df[col_images].tolist()
        self.annotations = df[col_annotations].tolist()
        self.device = device
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        image = cv.imread(self.images[idx])
        image = cv.resize(image, (256,256))
        image = image.reshape((3,256,256))
        image = torch.from_numpy(image).float().to(self.device)

        mask = cv.imread(self.annotations[idx],0)
        mask = cv.resize(mask, (256,256))
        mask = mask/255
        mask[mask>200]=255
        mask[mask<=200]=0
        mask = mask/255

        

        mask = torch.from_numpy(mask).long().to(self.device)
        return image, mask
    
