import os
import torch
import matplotlib.pyplot as plt
from dataloader import get_dataloader
from utils.utils import Adder, Timer, check_lr, draw_curve
from valid import _valid
from torchvision.transforms import functional as F
from tqdm import tqdm
import numpy as np
from torchsummary import summary
import argparse

####
# setting
####
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/local/SSD1/CV_Final_Group6/dataset/')
parser.add_argument('--model_save_dir', type=str, default='./results/KD_simple_v3/')
parser.add_argument('--hard_loss_ratio', type=float, default=0.8)
parser.add_argument('--soft_loss_ratio', type=float, default=0.2)
args = parser.parse_args()

# weights saving path
# model_save_dir = fr'./results/KD_simple_v3/'
# dataset path
# data_dir = '/local/SSD1/CV_Final_Group6/dataset/'
data_dir = args.data_dir
# model setting
teacher_name = 'model_b1'
student_model_name = 'CSPPeleeNet_small'
teacher_weight_path = fr'./results/origin_efficientb1/weights/best.pkl'
# no need to change
KD_model_name = 'KD_simple_v3'


os.makedirs(args.model_save_dir, exist_ok=True)

# train
learning_rate = 1e-3
batch_size = 32
num_worker = 8
lr_steps = [(x+1) * 4 for x in range(100//4)]
gamma = 0.7
num_epoch = 70
print_freq = 500

# loss
T = 10
# hard_loss_ratio = 0.8
# soft_loss_ratio = 0.2
loss_mse_ratio = 1
each_features_ratio = [0.2, 0.4, 0.6, 0.8]

#------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CE = torch.nn.CrossEntropyLoss()
MSE = torch.nn.MSELoss()

# load teacher model with pre_trained weight
print('loading teacher model with pretrain weitht.')
teachers = __import__(f'model.{teacher_name}', fromlist=[teacher_name])
teacher = teachers.build_net()
teacher.to('cuda')
teacher_state_dict = torch.load(teacher_weight_path)
teacher.load_state_dict(teacher_state_dict['model'])


# load student model
students = __import__(f'model.{student_model_name}', fromlist=[student_model_name])
student = students.build_net()

# load KD model (including student model)
print('loading KD model.')
models = __import__(f'model.{KD_model_name}', fromlist=[KD_model_name])
model = models.build_net(student)
model.to('cuda')
summary(model, (3,224, 224))  

# setting optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, gamma)

# data_loader                             
train_dataloader, val_dataloader = get_dataloader(data_dir, batch_size, num_worker, 'train')
max_iter = len(train_dataloader)
print('loading {max_iter} pairs image for training'.format(max_iter=max_iter))


#writer = SummaryWriter(os.path.join('runs',config.model_name))
epoch_adder = Adder()
iter_adder = Adder()
iter_ACC = Adder()
epoch_ACC = Adder()

epoch_timer = Timer('m')
iter_timer = Timer('m')


loss_train_metrices = []
loss_val_metrices = []
acc_train_metrices = []
acc_val_metrices = []

start_epoch = 0
best_acc = 0.0



for epoch_idx in range(start_epoch + 1, num_epoch + 1):
    epoch_timer.tic()
    iter_timer.tic()
    
    model.train()
    teacher.eval()
    
    for iter_idx, batch_data in enumerate(tqdm(train_dataloader)):
  
        img, label = batch_data
        img = img.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        # KD_simple_model predict
        pred, features = model(img)
        pred_class = torch.argmax(pred,1)
        
        # GT acc
        acc = (pred_class == label).sum()
        
        # teacher predict
        with torch.no_grad():
            teacher_pred, teacher_features = teacher(img, is_feat=True)
        
        # hard GT loss
        hard_loss = CE(pred, label)
        
        # soft loss
        ## Soften the student logits by applying softmax first and log() second
        soft_targets = torch.nn.functional.softmax(teacher_pred / T, dim=-1)
        soft_prob = torch.nn.functional.log_softmax(pred / T, dim=-1)
        ## Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)
        
        # features MSE
        loss_mse1 = MSE(features[0], teacher_features[0])
        loss_mse2 = MSE(features[1], teacher_features[1])
        loss_mse3 = MSE(features[2], teacher_features[2])
        loss_mse4 = MSE(features[3], teacher_features[3])
        
        loss_mse = loss_mse1*each_features_ratio[0] + loss_mse2*each_features_ratio[1]+loss_mse3*each_features_ratio[2] + loss_mse4*each_features_ratio[3]
                    
        # total loss
        loss = args.hard_loss_ratio*hard_loss + args.soft_loss_ratio*soft_loss + loss_mse_ratio*loss_mse
        
 
        loss.backward()
        optimizer.step()
        
        iter_adder(loss.item())
        epoch_adder(loss.item())
        iter_ACC(acc.item(), img.shape[0])
        epoch_ACC(acc.item(), img.shape[0])
        
        if (iter_idx + 1) % print_freq == 0:
            lr = check_lr(optimizer)
            print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss: %7.4f ACC: %.4f" % (iter_timer.toc(), epoch_idx,
                                                                         iter_idx + 1, max_iter, lr,
                                                                         iter_adder.average(), iter_ACC.average()))
            #writer.add_scalar('Loss', iter_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
            iter_timer.tic()
            iter_adder.reset()
            iter_ACC.reset()
    
    loss_train_metrices.append(epoch_adder.average())   
    acc_train_metrices.append(epoch_ACC.average())
           
    epoch_adder.reset()
    epoch_ACC.reset()
    scheduler.step()
    
    # valid
    loss, acc = _valid(student, val_dataloader)        
    acc_val_metrices.append(acc)
    loss_val_metrices.append(loss)  
    print('valid: %03d epoch \n Average loss %.4f ACC %.4f \n' % (epoch_idx, loss, acc))

    if acc > best_acc:
        save_name = os.path.join(args.model_save_dir, 'KD_simple_best.pkl')
        torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch_idx}, save_name)
        best_acc = acc


save_name = os.path.join(args.model_save_dir, 'Final.pkl')
torch.save({'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch_idx}, save_name)
            
# plot loss/ACC curve            
draw_curve(args.model_save_dir, loss_train_metrices, loss_val_metrices, 'Loss', start_epoch, num_epoch)            
draw_curve(args.model_save_dir, acc_train_metrices, acc_val_metrices, 'ACC', start_epoch, num_epoch)