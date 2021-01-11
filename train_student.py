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
from configparser import ConfigParser
from utils import fit_mel_size, AudioDataset, bce, loss_fn_kd, loss_fn_sp, bce_with_logits, train, val
from models import DCASE_Net, DCASE_Small, CNN_LSTM, MobileNetV2

parser  = ConfigParser()
parser.read('config.ini')

train_list = pd.read_csv('train.csv')
val_list = pd.read_csv('validate.csv')

train_dataset = AudioDataset(train_list, 'features/student', 'features/teacher')
train_load = DataLoader(train_dataset, batch_size = 8, shuffle=True, num_workers = 0)

val_dataset = AudioDataset(val_list, 'features/student', 'features/teacher')
val_load = DataLoader(val_dataset, batch_size = 8, shuffle=True, num_workers = 0)

# Hyperparameters
lr = parser.getfloat('student', 'LEARNING_RATE')
T = parser.getfloat('student', 'SOFTMAX_TEMPERATURE')
epochs = parser.getint('student', 'EPOCHS')

def getTeacherModel():
    teacher_model = parser.get('teacher', 'MODEL')

    if teacher_model=='dcase_net':
        teacher = DCASE_Net()
        teacher.load_state_dict(torch.load('weights/teacher', map_location=device))
        for param in teacher.parameters():
            param.requires_grad = False

    return teacher

def getStudentModel():
    student_model = parser.get('student', 'MODEL')
    
    if student_model=='dcase_small':
        student = DCASE_Small()
        student = student.cuda()

    elif student_model=='cnn_lstm':
        student = CNN_LSTM()
        student = student.cuda()

    elif student_model=='mobilenetv2':
        width_mult = parser.getint('models', 'width_mult')
        student = MobileNetV2(width_mult, 8)

    return student

teacher = getTeacherModel()
student = getStudentModel()
teacher = teacher.cuda()
student = student.cuda()

# Baseline
max_acc_baseline = 0.0
for i in range(epochs):
    
    print("Training epoch {}".format(i))
    train(student, teacher, T, train_load, 1.5, 0, 0, lr)
    print("Validating epoch {}".format(i))
    micro_auprc = val(student, val_load, i)
    print("Micro AUPRC: {}".format(micro_auprc))
  
    if(micro_auprc>max_acc_baseline):
        torch.save(student.state_dict(), 'weights/dcase_small_baseline')
        max_acc_baseline = micro_auprc

student = getStudentModel()
student = student.cuda()

# KD
max_acc_kd = 0.0
for i in range(epochs):
    
    print("Training epoch {}".format(i))
    train(student, teacher, T, train_load, 1.5, 1.5, 0, lr)
    print("Validating epoch {}".format(i))
    micro_auprc = val(student, val_load, i)
    print("Micro AUPRC: {}".format(micro_auprc))
  
    if(micro_auprc>max_acc_kd):
        torch.save(student.state_dict(), 'weights/dcase_small_kd')
        max_acc_kd = micro_auprc

student = getStudentModel()
student = student.cuda()

# SP
max_acc_sp = 0.0
for i in range(epochs):
    
    print("Training epoch {}".format(i))
    train(student, teacher, T, train_load, 1.5, 1.5, 10, lr)
    print("Validating epoch {}".format(i))
    micro_auprc = val(student, val_load, i)
    print("Micro AUPRC: {}".format(micro_auprc))
  
    if(micro_auprc>max_acc_sp):
        torch.save(student.state_dict(), 'weights/dcase_small_sp')
        max_acc_sp = micro_auprc

print("Baseline micro AUPRC: ", max_acc_baseline)
print("KD AUPRC: ", max_acc_kd)
print("SP AUPRC: ", max_acc_sp)