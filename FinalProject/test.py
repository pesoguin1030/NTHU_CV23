import torch
from torchvision.transforms import functional as F
import os
import numpy as np
import pandas as pd
from dataloader import get_dataloader


def _mytest(model, config):
    print('start test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_pretrained = os.path.join(config.model_save_dir, 'best.pkl')
    state_dict = torch.load(model_pretrained, map_location=torch.device(device))
    model.load_state_dict(state_dict['model'])       
  
    dataloader = get_dataloader(config.data_dir, 1, config.num_worker, 'test')
    
    pred_labels = []   
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(dataloader):
        
            img, names = batch_data
            img = img.to(device)
            img_idx = int(names[0].split('.')[0])
            pred = model(img)
            pred_class = int(torch.argmax(pred,1)[0])
            
            pred_labels.append([img_idx, pred_class])
            
            if idx % 300 == 0:
                print('\r%05d'%idx, end=' ')
                
        df_pred = pd.DataFrame(pred_labels)
        df_pred.columns = ['id', 'predict']
        df_pred = df_pred.sort_values('id')
        df_pred.to_csv(f'{config.result_dir}/test.csv', index=False)
            
    

