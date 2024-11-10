import torch
from torchvision.transforms import functional as F
import os
import numpy as np
import pandas as pd
from dataloader import get_dataloader
from torchsummary import summary
####
# setting
####

# path
KD_model_weight_path = fr'./results/CSPsmall_KD_v2_h0.0_s1.0/weights/KD_simple_best.pkl'
data_dir = fr'/local/SSD1/CV_Final_Group6/dataset/'
result_dir = fr'./results/CSPsmall_KD_v2_h0.0_s1.0/'
# loading setting
num_worker = 8
student_model_name = 'CSPPeleeNet_small'
KD_model_name = 'KD_simple_v2'



print('start test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create KD model and student model
students = __import__(f'model.{student_model_name}', fromlist=[student_model_name])
student = students.build_net()
models = __import__(f'model.{KD_model_name}', fromlist=[KD_model_name])
KD_model = models.build_net(student)
KD_model.to(device)

# loading weights
state_dict = torch.load(KD_model_weight_path, map_location=torch.device(device))
print('loading the whole KD model weights')
KD_model.load_state_dict(state_dict['model'])  
print('divied the student model from KD model')
model = KD_model.student
summary(model, (3,224, 224))      

dataloader = get_dataloader(data_dir, 1, num_worker, 'test')

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
    df_pred.to_csv(f'{result_dir}/test.csv', index=False)
            
    

