import sys
import os
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader    
import torchsummary
from torch.utils.tensorboard import SummaryWriter
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm, trange

class DCASE_Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layer1_conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.layer2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3_conv2 = nn.Sequential(nn.Conv2d(64, 128,kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.layer4_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5_conv3_1 = nn.Sequential(nn.Conv2d(128, 256,kernel_size=3, stride=1,padding=1), nn.ReLU())
        self.layer6_conv3_2 = nn.Sequential(nn.Conv2d(256, 256,kernel_size=3, stride=1,padding=1), nn.ReLU())
        self.layer7_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer8_conv4_1 = nn.Sequential(nn.Conv2d(256, 512,kernel_size=3, stride=1,padding=1), nn.ReLU())
        self.layer9_conv4_2 = nn.Sequential(nn.Conv2d(512, 512,kernel_size=3, stride=1,padding=1), nn.ReLU())
        self.new_fc1 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU())
        self.new_fc2 = nn.Sequential(nn.Linear(2048, 128), nn.ReLU())
        self.final= nn.Sequential(nn.Linear(128, 8))

    def forward(self, x):
        x = self.layer1_conv1(x)
        out = self.layer2_pool1(x)
        out = self.layer3_conv2(out)
        out = self.layer4_pool2(out)
        out = self.layer5_conv3_1(out)
        out = self.layer6_conv3_2(out)
        out = self.layer7_pool3(out)
        out = self.layer8_conv4_1(out)
        out = self.layer9_conv4_2(out)
        out1 = out

        # maxpooling
        out = torch.max(out, dim=2)[0]
        out = out.view(out.size(0),-1)
        out = self.new_fc1(out)
        out = self.new_fc2(out)
        out = self.final(out)
        
        return out, out1

class DCASE_Small(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layer1_conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.layer2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3_conv2 = nn.Sequential(nn.Conv2d(16, 32,kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.layer4_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5_conv3_1 = nn.Sequential(nn.Conv2d(32, 64,kernel_size=3, stride=1,padding=1), nn.ReLU())
        self.layer6_conv3_2 = nn.Sequential(nn.Conv2d(64, 64,kernel_size=3, stride=1,padding=1), nn.ReLU())
        self.layer7_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer8_conv4_1 = nn.Sequential(nn.Conv2d(64, 128,kernel_size=3, stride=1,padding=1), nn.ReLU())
        self.layer9_conv4_2 = nn.Sequential(nn.Conv2d(128, 128,kernel_size=3, stride=1,padding=1), nn.ReLU())
        self.new_fc1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.new_fc2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.final= nn.Sequential(nn.Linear(64, 8))

    def forward(self, x):
        x = self.layer1_conv1(x)
        out = self.layer2_pool1(x)
        out = self.layer3_conv2(out)
        out = self.layer4_pool2(out)
        out = self.layer5_conv3_1(out)
        out = self.layer6_conv3_2(out)
        out = self.layer7_pool3(out)
        out = self.layer8_conv4_1(out)
        out = self.layer9_conv4_2(out)
        out1 = out

        # maxpooling
        out = torch.max(out, dim=2)[0]
        out = out.view(out.size(0),-1)
        # print(out.shape)
        out = self.new_fc1(out)
        out = self.new_fc2(out)
        out = self.final(out)
        
        return out, out1

class CNN_LSTM(nn.Module):
    
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        
        self.conv = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.lstm = nn.LSTM(32*8, 64, 1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 8))

    def forward(self, x):
        out = self.conv(x)
        out1 = out
        # out = out.view(out.shape[0], -1)

        out = out.view(out.shape[0], 497, 32*8)
        out,_ = self.lstm(out)
        out = out[:,-1,:]
        out = self.fc(out)
        return out, out1

class MobileNetV2(nn.Module):

    def __init__(self, width_mult, num_classes):

        super().__init__()

        self.bw2col = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 10, 1, padding=0), nn.ReLU(),
            nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())

        self.mv2 = torchvision.models.mobilenet_v2(pretrained=False, width_mult=width_mult)

        self.final = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.bw2col(x)
        x = self.mv2.features(x)
        y = x
        x = x.max(dim=-1)[0].max(dim=-1)[0]
        x = x.view(x.shape[0], -1)
        x = self.final(x)
        return x, y      