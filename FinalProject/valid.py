import torch
from torchvision.transforms import functional as F
from utils.utils import Adder
import os
import numpy as np


def _valid(model, val_dataloader):
    print('start valid')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_adder = Adder()
    ACC_adder = Adder()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(val_dataloader):
        
            img, label = batch_data
            img = img.to(device)
            label = label.to(device)
            
            pred = model(img)
            pred_class = torch.argmax(pred,1)
            
            loss = criterion(pred, label)
            acc = (pred_class == label).sum()
     
            ACC_adder(acc.item(), img.shape[0])
            loss_adder(loss.item())
            
    model.train()
    print('-'*30)
    return loss_adder.average(), ACC_adder.average()

