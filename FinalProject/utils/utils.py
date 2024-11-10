import time
import numpy as np
import matplotlib.pyplot as plt
import os

class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num, count=1):
        self.count += count
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


class EvalTimer(object):
    def __init__(self):
        self.num = 0
        self.total = 0
        self.tmp = 0

    def tic(self):
        self.tmp = time.time()

    def toc(self):
        self.total = time.time() - self.tmp
        self.num += 1

    def print_time(self):
        print('Total %d images, take %f  FPS: %f'%(self.num, self.total, self.total/self.num))

# Plot the loss/ACC curve against epoch
def draw_curve(save_root, train_metrices, val_metrices, title, start_epoch, max_epoch):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    ax.set_title(title)
    ax.plot(range(start_epoch, max_epoch), train_metrices, label='Train')
    ax.plot(range(start_epoch, max_epoch), val_metrices, label='Valid')
    ax.legend()
    fig.savefig(os.path.join(save_root, f'{title}.jpg'))
    plt.close(fig)


class WarmUpScheduler(object):
    def __init__(self, optimizer, base_lr, target_lr, warm_up_iters):
        self.base_lr = base_lr
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.warm_up_iters = warm_up_iters
        self.iters = 0

    def step(self):
        self.iters += 1
        lr = ((self.target_lr - self.base_lr) / self.warm_up_iters) * self.iters + self.base_lr
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr


class PolynomialScheduler(object):
    def __init__(self, optimizer, base_lr, target_lr, power, max_epoch):
        self.base_lr = base_lr
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.power = power
        self.max_epoch = max_epoch
        self.epoch = 0

    def step(self):
        lr = (self.base_lr - self.target_lr) * (1 - self.epoch/self.max_epoch)**self.power + self.target_lr
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr
        self.epoch += 1
