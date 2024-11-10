"""
#This is the KD simple model, version 2 (v2)
#The difference between KD_simple_model and v2 is the number of feature layers 
used to calculate distillation (mse) loss.
#v2 version will take multiple intermediate layers to calculate distillation loss,
while KD_simple will only take the last layer (after avgpool)
"""

import torch
from torch import nn
import torch.nn.functional as F


class linear(nn.Sequential):
    def __init__(self, inp, oup):
        super(linear, self).__init__(
            nn.Linear(inp, oup, bias=False),
            nn.SiLU()
        )

class Conv1x1(nn.Sequential):
    def __init__(self, inp, oup):
        super(Conv1x1, self).__init__(
            nn.Conv2d(inp, oup, 1, bias=False),
            nn.BatchNorm2d(oup), # I am not sure the effect of batch normalization
            nn.SiLU()
        )      

class KD_simple_v2(nn.Module):
    def __init__(self, student, in_features, out_features):
        super(KD_simple_v2, self).__init__()
        self.conv1 = Conv1x1(in_features[0], out_features[0])
        # self.conv2 = nn.Sequential(
                        # nn.Upsample(scale_factor=(2,2)),    
                        # Conv1x1(in_features[1], out_features[1])
                     # )   
        self.conv2 = Conv1x1(in_features[1], out_features[1])        
        self.conv3 = Conv1x1(in_features[2], out_features[2])
        self.linear = linear(in_features[-1], out_features[-1])
        
        self.student = student
        
    def forward(self, x):
        pred, features = self.student.forward(x, is_feat=True)
        feat1, feat2, feat3, feat = features
        feature1 = self.conv1(feat1)
        feature2 = self.conv2(feat2)
        # print(feature2.size())
        feature3 = self.conv3(feat3)
        feature = self.linear(feat)
        return pred, [feature1, feature2, feature3, feature]
            


def build_net(student=None):
    assert student != None, 'no student model loading.'
    #2.5M
    # model = KD_simple_v2(student, in_features=[128, 256, 512, 704] , out_features=[40, 112, 320, 1280])
    #1.39M
    model = KD_simple_v2(student, in_features=[96, 192, 384, 480] , out_features=[40, 112, 320, 1280])
    return model