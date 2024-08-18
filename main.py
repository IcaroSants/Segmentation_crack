from torch.utils.data import random_split, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, deeplabv3
from data_loader import CustomImageDataset
from train_model import fit
import pandas as pd
import glob 
import cv2 as cv
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier= deeplabv3.DeepLabHead(2048,2)

model.to(device)

optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

df = pd.read_csv('dataset.csv')
dataset = CustomImageDataset(df, 'Images','Mask', device)

train, test, val = random_split(dataset,[0.8,0.1,0.1])

train = DataLoader(train, batch_size=5, shuffle=True)
test = DataLoader(test, batch_size=5, shuffle=True)
val = DataLoader(val, batch_size=5, shuffle=True)


fit(model,train,val,criterion=criterion, optimizer=optimizer, epochs=1)
torch.save(model.state_dict(), "weights.pt")

