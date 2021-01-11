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


# Binary cross entropy loss
def bce(outputs, labels):
    return nn.BCELoss()(outputs, labels)

# Knowledge distillation loss function
def loss_fn_kd(student, teacher, T):
    temp_student = nn.Sigmoid()(student/T)
    temp_teacher = nn.Sigmoid()(teacher/T)
    first_term = bce(temp_student, temp_teacher)
    return first_term

# Similarity preserving loss function
def loss_fn_sp(f_s, f_t):
    
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    G_s = G_s / G_s.norm(2, dim=1)
    # G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_t = G_t / G_t.norm(2, dim=1)
    # G_t = torch.nn.functional.normalize(G_t)
    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss

# BCE With Logits loss
def bce_with_logits(output, target):
    return nn.BCEWithLogitsLoss()(output, target)