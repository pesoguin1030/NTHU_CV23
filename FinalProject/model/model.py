import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary

class EFFICIENTNET_B0(nn.Module):
    def __init__(self, num_class=525, pretrain=True):
        super(EFFICIENTNET_B0, self).__init__()
        if pretrain:
            efficient = models.efficientnet_b0(weights='DEFAULT')
        else:
            efficient = models.efficientnet_b0()
        
        # freeze
        self._set_parameter_requires_grad(efficient.features, 5)
        
        efficient.classifier = nn.Sequential(
                                        nn.Dropout(p=0.3),
                                        nn.Linear(in_features=1280, out_features=num_class, bias=True),
                                    )
        self.efficient = efficient
        
    # freeze the parameter of the first 17 layers of pretrained vgg16 (3 times downsample)
    def _set_parameter_requires_grad(self, model, pretrain_block):
        if not pretrain_block == 0:
            for idx in range(0,pretrain_block):
                for param in model[idx].parameters():
                    param.requires_grad = False
    
    
    def forward(self, x, is_feat=True):     
        feat1 = self.efficient.features[3](self.efficient.features[0:3](x))
        feat2 = self.efficient.features[5](self.efficient.features[4](feat1))
        feat3 = self.efficient.features[7](self.efficient.features[6](feat2))
        feat = self.efficient.features[8](feat3)
        feat = self.efficient.avgpool(feat)
        feat = nn.Flatten()(feat)
        pred = self.efficient.classifier(feat)
        if not is_feat:
            return pred
        else: # return feature
            return pred, [feat1, feat2, feat3, feat]
            
        
# if __name__ == '__main__': 
    # model = EFFICIENTNET_B0()
    # print(model)
    # model.to('cuda')
    # summary(model, (3,224, 224))  

    
def build_net():
    return EFFICIENTNET_B0()