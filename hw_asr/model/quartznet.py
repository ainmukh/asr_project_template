from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from hw_asr.base import BaseModel


class Conv(nn.Module):
    def __init__(self,
                 in_channels: int = 128, out_channels: int = 256,
                 kernel_size: int = 33,
                 stride: int = 1, padding: int = 0, dilation: int = 1):
        super(Conv, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        # self.relu = nn.ReLU()

    def forward(self, spectrogram):
        # print('input size =', spectrogram.size())
        x = self.depthwise(spectrogram)
        # print('after depthwise =', x.size())
        x = self.pointwise(x)
        # print('after pointwise =', x.size())
        x = self.bn(x)
        # x = self.relu(x)
        return x


class Block(nn.Module):
    def __init__(self, r: int = 5,
                 in_channels: int = 256, out_channels: int = 256, kernel_size: int = 33):
        super(Block, self).__init__()
        padding = (kernel_size - 1) // 2
        self.base1 = Conv(in_channels, out_channels, kernel_size, padding=padding)
        self.bases = []
        for i in range(1, r):
            self.bases.append(nn.ReLU())
            self.bases.append(Conv(out_channels, out_channels, kernel_size, padding=padding))
        self.bases = nn.Sequential(*self.bases)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, spectrogram):
        # print('input size =', spectrogram.size())
        res = self.residual(spectrogram)
        x = self.base1(spectrogram)
        # print('after 1st layer =', x.size())
        x = self.bases(x)
        # print('output size =', x.size(), '\n')
        x = F.relu(x + res)
        return x


class QuartzNet(BaseModel):
    def __init__(self, n_feats, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.conv1 = Conv(n_feats, stride=2, padding=(33 - 1) // 2)
        self.blocks = nn.Sequential(
            Block(),
            Block(kernel_size=39),
            Block(out_channels=512, kernel_size=51),
            Block(in_channels=512, out_channels=512, kernel_size=63),
            Block(in_channels=512, out_channels=512, kernel_size=75)
        )
        self.conv2 = Conv(512, 512, 87, 1, 86, 2)
        self.conv3 = Conv(512, 1024, 1)
        self.conv4 = Conv(1024, n_class, 1)
        self.conv4 = nn.Conv1d(1024, n_class, kernel_size=1, bias=False)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal(p)
            else:
                nn.init.normal_(p)

    def forward(self, spectrogram, *args, **kwargs):
        x = F.relu(self.conv1(spectrogram))
        # print('conv1 =', x.size())
        x = self.blocks(x)
        x = F.relu(self.conv2(x))
        # print('conv2 =', x.size())
        x = F.relu(self.conv3(x))
        # print('conv3 =', x.size())
        x = self.conv4(x)
        # print('conv4 =', x.size())
        return x.transpose(1, 2)

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
