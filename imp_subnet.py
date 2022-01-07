import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.module_util as mutil
import modules.Unet_common as common
import functools
import config as c
from non_local_dot_product import NONLocalBlock2D


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()

        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.senet = SELayer(channel=input + 4 * 32)
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x = torch.cat((x, x1, x2, x3, x4), 1)
        x = self.senet(x)
        x5 = self.conv5(x)
        return x5


class ResidualDenseBlock_out2(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out2, self).__init__()

        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x = torch.cat((x, x1, x2, x3, x4), 1)
        x5 = self.conv5(x)
        return x5


# Dense connection
class PrepareBlock(nn.Module):
    def __init__(self, input, bias=True):
        super(PrepareBlock, self).__init__()

        self.conv1 = nn.Conv2d(input, 64, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 64, 64, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 64, 64, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.senet = SELayer(channel=input + 2 * 64)
        mutil.initialize_weights([self.conv3], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x = torch.cat((x, x1, x2), 1)
        x = self.senet(x)
        x3 = self.conv3(x)
        return x3


class ImpMapBlock(nn.Module):
    def __init__(self):
        super(ImpMapBlock, self).__init__()
        self.rrdb = ResidualDenseBlock_out(input=64 + 64 + 64, output=3)
        self.s_cover = PrepareBlock(input=3)
        self.s_secret = PrepareBlock(input=3)
        self.s_steg = PrepareBlock(input=3)
        self.nonlocal_block = NONLocalBlock2D(3, sub_sample=True, bn_layer=True)

    def forward(self, cover, secret, steg):

        x_cover = self.s_cover(cover)
        x_steg = self.s_steg(steg)
        x_secret = self.s_secret(secret)
        x = torch.cat((x_cover, x_steg), 1)
        x = torch.cat((x, x_secret), 1)
        x = self.rrdb(x)

        return x


class ImpMapBlock_Nonlocal(nn.Module):
    def __init__(self):
        super(ImpMapBlock_Nonlocal, self).__init__()
        self.rrdb = ResidualDenseBlock_out2(input=64 + 64 + 64, output=3)
        self.s_cover = PrepareBlock(input=3)
        self.s_secret = PrepareBlock(input=3)
        self.s_steg = PrepareBlock(input=3)
        self.nonlocal_block = NONLocalBlock2D(3, sub_sample=True, bn_layer=True)

    def forward(self, cover, secret, steg):

        x_cover = self.s_cover(cover)
        x_steg = self.s_steg(steg)
        x_secret = self.s_secret(secret)
        x = torch.cat((x_cover, x_steg), 1)
        x = torch.cat((x, x_secret), 1)
        x = self.rrdb(x)
        x = self.nonlocal_block(x)
        return x

