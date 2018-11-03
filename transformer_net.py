# coding:utf8

import torch as t
from torch import nn
import numpy as np


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 32, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 3, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(32, 64, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(64, 32, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(32, 3, 3, padding=1)
    
    def forward(self, x):
        t = self.conv1(x)
        t = self.relu1(t)
        t = self.conv2(t)
        t = self.relu2(t)
        t = self.conv3(t)
        t = self.relu3(t)
        t = self.conv4(t)
        t = self.relu4(t) + x
        
        t = self.conv5(t)
        t = self.relu5(t)
        t = self.conv6(t)
        t = self.relu6(t)
        t = self.conv7(t)
        t = self.relu7(t)
        t = self.conv8(t) + x
        return t


class Transformer_2layer(nn.Module):
    def __init__(self):
        super(Transformer_2layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 32, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 3, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        t = self.conv1(x)
        t = self.relu1(t)
        t = self.conv2(t)
        t = self.relu2(t)
        t = self.conv3(t)
        t = self.relu3(t)
        t = self.conv4(t)
        t = self.relu4(t) + x
        return t
