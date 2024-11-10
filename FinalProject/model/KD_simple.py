import torch
from torch import nn
import torch.nn.functional as F

class linear(nn.Sequential):
    def __init__(self, inp, oup):
        super(linear, self).__init__(
            nn.Linear(inp, oup, bias=False),
            # nn.BatchNorm2d(oup),
            nn.SiLU()
        )
        

class KD_simple(nn.Module):
    def __init__(self, student, in_features, out_features):
        super(KD_simple, self).__init__()
        self.linear = linear(in_features, out_features)
        self.student = student
        
    def forward(self, x):
        pred, features = self.student.forward(x, is_feat=True)
        feat = features[-1]
        features = self.linear(feat)
        return pred, features


def build_net(student=None):
    assert student != None, 'no student model loading.'
    #1.39M
    model = KD_simple(student, in_features=480 , out_features=1280)
    #2.5M
    # model = KD_simple(student, in_features=704 , out_features=1280)
    return model