import torch.nn as nn
import torch
import math
from torch.nn import functional as F
import copy
from functools import partial
from typing import Optional, Callable

import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"



# 定义CConv类，将Unet内多次用到的几个相同步骤组合在一起成一个网络，避免重复代码太多
class CConv(nn.Module):
    # 定义网络结构
    def __init__(self, in_ch, out_ch):
        super(CConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    # 重写父类forward方法
    def forward(self, t):
        t = self.conv1(t)
        t = self.bn1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = self.bn2(t)
        output = F.relu(t)
        return output


# 纯Unet
class Unet(nn.Module):
    # 定义网络结构共9层，四次下采样，四次上采样
    def __init__(self, input_channels, output_channels):
        super(Unet, self).__init__()
        self.conv1 = CConv(in_ch=input_channels, out_ch=64)
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = CConv(in_ch=64, out_ch=128)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = CConv(in_ch=128, out_ch=256)
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = CConv(in_ch=256, out_ch=512)
        self.down4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = CConv(in_ch=512, out_ch=1024)

        self.bn1 = nn.BatchNorm2d(num_features=1024)
        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv6 = CConv(in_ch=1024, out_ch=512)

        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv7 = CConv(in_ch=512, out_ch=256)

        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv8 = CConv(in_ch=256, out_ch=128)

        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv9 = CConv(in_ch=128, out_ch=64)

        self.conv10 = nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=1)


    # 重写父类forward方法
    def forward(self, t):
        # layer1
        c1 = self.conv1(t)
        t = self.down1(c1)

        # layer2
        c2 = self.conv2(t)
        t = self.down2(c2)

        # layer3
        c3 = self.conv3(t)
        t = self.down3(c3)

        # layer4
        c4 = self.conv4(t)
        t = self.down4(c4)

        # layer5
        c5 = self.conv5(t)

        d1 = self.up1(c5)

        # layer6
        t = torch.cat([d1, c4], dim=1)
        t = self.conv6(t)
        d2 = self.up2(t)

        # layer7
        t = torch.cat([d2, c3], dim=1)
        t = self.conv7(t)
        d3 = self.up3(t)

        # layer8
        t = torch.cat([d3, c2], dim=1)
        t = self.conv8(t)
        d4 = self.up4(t)

        # layer9
        t = torch.cat([d4, c1], dim=1)
        t = self.conv9(t)
        t = self.conv10(t)
        out = torch.sigmoid(t)
        return out


# 测试SE模块代码
if __name__ == '__main__':

    x = torch.rand(16, 3, 224, 224).to('cuda')
    model = Unet(3, 1).to('cuda')
    output = model(x)
    print(output.shape)
