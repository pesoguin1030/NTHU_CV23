import os
import torch
import matplotlib.pyplot as plt
from dataloader import get_dataloader
from utils.utils import Adder, Timer, check_lr, draw_curve
from valid import _valid
from torchvision.transforms import functional as F
from tqdm import tqdm
import numpy as np



def _train(model, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)
    train_dataloader, val_dataloader = get_dataloader(config.data_dir, config.batch_size, config.num_worker, 'train')
    max_iter = len(train_dataloader)
    print('loading {max_iter} pairs image for training'.format(max_iter=max_iter))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_steps, config.gamma)

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
    
    if config.resume_train != None:
        state_dict = torch.load(config.resume_train)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        start_epoch = state_dict['epoch']
    # print("No previous data... Started from scratch ... \n")
    
    
    for epoch_idx in range(start_epoch + 1, config.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(tqdm(train_dataloader)):
      
            img, label = batch_data
            img = img.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            pred = model(img)
            pred_class = torch.argmax(pred,1)
            
            loss = criterion(pred, label)
            acc = (pred_class == label).sum()
            
     
            loss.backward()
            optimizer.step()
            
            iter_adder(loss.item())
            epoch_adder(loss.item())
            iter_ACC(acc.item(), img.shape[0])
            epoch_ACC(acc.item(), img.shape[0])
            
            if (iter_idx + 1) % config.print_freq == 0:
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
        loss, acc = _valid(model, val_dataloader)        
        acc_val_metrices.append(acc)
        loss_val_metrices.append(loss)  
        print('valid: %03d epoch \n Average loss %.4f ACC %.4f \n' % (epoch_idx, loss, acc))

        if acc > best_acc:
            save_name = os.path.join(config.model_save_dir, 'best.pkl')
            torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, save_name)
            best_acc = acc

    
    save_name = os.path.join(config.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch_idx}, save_name)
                
    # plot loss/ACC curve            
    draw_curve(config.model_save_dir, loss_train_metrices, loss_val_metrices, 'Loss', start_epoch, config.num_epoch)            
    draw_curve(config.model_save_dir, acc_train_metrices, acc_val_metrices, 'ACC', start_epoch, config.num_epoch)