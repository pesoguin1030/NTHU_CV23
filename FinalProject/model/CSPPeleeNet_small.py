import torch
import torch.nn as nn
from torchsummary import summary

class Conv3x3(nn.Sequential):
    def __init__(self, inp, oup, stride, pad, relu=True):
        if relu:
            relu = nn.ReLU(True)
        else:
            relu = nn.Identity()
        super(Conv3x3, self).__init__(
            nn.Conv2d(inp, oup, 3, stride, pad, bias=False),
            nn.BatchNorm2d(oup),
            relu
        )


class Conv1x1(nn.Sequential):
    def __init__(self, inp, oup, stride, pad, relu=True):
        if relu:
            relu = nn.ReLU(True)
        else:
            relu = nn.Identity()
        super(Conv1x1, self).__init__(
            nn.Conv2d(inp, oup, 1, stride, pad, bias=False),
            nn.BatchNorm2d(oup),
            relu
        )


class StemBlock(nn.Module):
    def __init__(self, inp):
        super(StemBlock, self).__init__()
        self.conv1 = Conv3x3(inp, 32, 2, 1)
        self.conv2 = nn.Sequential(
            Conv1x1(32, 16, 1, 0),
            Conv3x3(16, 32, 2, 1)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv1x1(64, 32, 1, 0)

    def forward(self, x):
        feat = self.conv1(x)
        feat_ = self.conv2(feat)
        feat = torch.cat([self.max_pool(feat), feat_], dim=1)
        feat = self.conv3(feat)
        return feat


class TwoWayDenseBlock(nn.Module):
    def __init__(self, inp, growth_rate, inter_ch):
        super(TwoWayDenseBlock, self).__init__()
        self.left = nn.Sequential(
            Conv1x1(inp, inter_ch, 1, 0),
            Conv3x3(inter_ch, growth_rate//2, 1, 1)
        )
        self.right = nn.Sequential(
            Conv1x1(inp, inter_ch, 1, 0),
            Conv3x3(inter_ch, growth_rate//2, 1, 1),
            Conv3x3(growth_rate//2, growth_rate//2, 1, 1)
        )

    def forward(self, x):
        feat_l = self.left(x)
        feat_r = self.right(x)
        feat = torch.cat([x, feat_l, feat_r], dim=1)
        return feat


class TransitionBlock(nn.Sequential):
    def __init__(self, inp, pool=True):
        if pool:
            pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            pool = nn.Identity()
        super(TransitionBlock, self).__init__(
            Conv1x1(inp, inp, 1, 0),
            pool
        )


class DenseStage(nn.Module):
    def __init__(self, inp, nblock, bwidth, growth_rate, pool):
        super(DenseStage, self).__init__()
        current_ch = inp
        inter_ch = int(growth_rate // 2 * bwidth / 4) * 4
        dense_branch = nn.Sequential()
        for i in range(nblock):
            dense_branch.add_module("dense{}".format(
                i+1), TwoWayDenseBlock(current_ch, growth_rate, inter_ch))
            current_ch += growth_rate
        dense_branch.add_module(
            "transition1", TransitionBlock(current_ch, pool=pool))
        self.dense_branch = dense_branch

    def forward(self, x):
        return self.dense_branch(x)


class CSPDenseStage(DenseStage):
    def __init__(self, inp, nblock, bwidth, growth_rate, pool, partial_ratio):
        split_ch = int(inp * partial_ratio)
        super(CSPDenseStage, self).__init__(
            split_ch, nblock, bwidth, growth_rate, False)
        self.split_ch = split_ch
        current_ch = inp + (growth_rate * nblock)
        self.transition2 = TransitionBlock(current_ch, pool=pool)

    def forward(self, x):
        x1 = x[:, :self.split_ch, ...]
        x2 = x[:, self.split_ch:, ...]
        feat1 = self.dense_branch(x1)
        feat = torch.cat([x2, feat1], dim=1)
        feat = self.transition2(feat)
        return feat


class PeleeNet(nn.Module):
    # origin nblocks=[3, 4, 8, 6], bottleneck_widths=[1, 2, 4, 4]
    def __init__(self, inp=3, nclass=525, growth_rate=32, nblocks=[2, 3, 6, 3],
                 bottleneck_widths=[1, 2, 4, 4], partial_ratio=1.0):
        super(PeleeNet, self).__init__()

        self.stem = StemBlock(inp)
        current_ch = 32
        stages = nn.Sequential()
        pool = True
        assert len(nblocks) == len(bottleneck_widths)
        for i, (nblock, bwidth) in enumerate(zip(nblocks, bottleneck_widths)):
            if (i+1) == len(nblocks):
                pool = False
            if partial_ratio < 1.0:
                stage = CSPDenseStage(
                    current_ch, nblock, bwidth, growth_rate, pool, partial_ratio)
            else:
                stage = DenseStage(current_ch, nblock,
                                   bwidth, growth_rate, pool)
            stages.add_module("stage{}".format(i+1), stage)
            current_ch += growth_rate * nblock
        self.stages = stages
        self.classifier = nn.Linear(current_ch, nclass)

    def forward(self, x, is_feat=False):
        feat0 = self.stem(x)
        # feat1.size=(96,28,28)
        feat1 = self.stages[0](feat0)
        # feat2.size=(192,14,14)
        feat2 = self.stages[1](feat1)
        # feat3.size=(384,7,7)
        feat3 = self.stages[2](feat2)
        feat = self.stages[3](feat3)
        # feat.size=(480,)        
        feat = torch.mean(feat, dim=[2, 3])  # GAP
        pred = self.classifier(feat)
        if not is_feat:
            return pred
        else:  # return feature
            return pred, [feat1, feat2, feat3, feat]
            

def build_net():
    return PeleeNet(nclass=525, partial_ratio=0.5)   
    
# if __name__ == '__main__':
    # model = PeleeNet(nclass=525, partial_ratio=0.5)
    # print(model)
    # model.to('cuda')
    # summary(model, (3,224, 224))        