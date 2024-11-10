import os
import torch
import argparse
from torch.backends import cudnn
from train import _train
from test import _mytest
from torchsummary import summary
import numpy as np
import random

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main(config):
    cudnn.benchmark = True
    set_seed(77)
    # if not os.path.exists('results/'):
        # os.makedirs(config.model_save_dir)
    # if not os.path.exists('results/' + config.model_name + '/'):
        # os.makedirs('results/' + config.model_name + '/')
    if config.mode == 'train':
        if not os.path.exists(config.model_save_dir):
            os.makedirs(config.model_save_dir)
    if config.mode == 'test':
        if not os.path.exists(config.result_dir):
            os.makedirs(config.result_dir)

    models = __import__(f'model.{config.model_name}', fromlist=[config.model_name])
    model = models.build_net()
    
    if torch.cuda.is_available():
        model.cuda()
        summary(model, (3,224,224))
    if config.mode == 'train':
        _train(model, config)

    elif config.mode == 'test':
        _mytest(model, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--data_dir', type=str, default='/local/SSD1/CV_Final_Group6/dataset/')
        
    # Train
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 5 for x in range(100//5)])
    parser.add_argument('--store_opt', type=bool, default=True)
    parser.add_argument('--resume_train', type=str, default=None)

    # Test
    parser.add_argument('--test_model', type=str, default='results/weights')
    parser.add_argument('--mode', type=str, default='train')

    # Save dirctory
    parser.add_argument('--model_save_dir', type=str, default=os.path.join('results', 'weights/'))
    parser.add_argument('--result_dir', type=str, default=os.path.join('results', 'test/'))
    
    args = parser.parse_args()
     
    print(args)
    main(args)
