import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ConvBlock


class BabyUnet(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, width=16):
        super(BabyUnet, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.up1 = lambda x: F.interpolate(x, mode='bilinear', scale_factor=2, align_corners=False)
        self.up2 = lambda x: F.interpolate(x, mode='bilinear', scale_factor=2, align_corners=False)

        self.conv1 = ConvBlock(n_channel_in, width)
        self.conv2 = ConvBlock(width, 2*width)

        self.conv3 = ConvBlock(2*width, 2*width)

        self.conv4 = ConvBlock(4*width, 2*width)
        self.conv5 = ConvBlock(3*width, width)

        self.conv6 = nn.Conv2d(width, n_channel_out, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        x = self.pool1(c1)
        c2 = self.conv2(x)
        x = self.pool2(c2)
        self.conv3(x)

        x = self.up1(x)
        x = torch.cat([x, c2], 1)
        x = self.conv4(x)
        x = self.up2(x)
        x = torch.cat([x, c1], 1)
        x = self.conv5(x)
        x = self.conv6(x)
        return x
