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
from loss_fn import bce, bce_with_logits, loss_fn_sp, loss_fn_kd


# Function to ensure that the audio files are of the same size
def fit_mel_size(mel, mel_frames=998):
    if mel.shape[0]<mel_frames:
        padding_len = mel_frames-mel.shape[0]
        zero_pad = np.zeros((padding_len, mel.shape[1]))
        mel = np.vstack((mel, zero_pad))
    elif mel.shape[0]>mel_frames:
        mel = mel[:mel_frames,:]
    return mel

class AudioDataset(Dataset):

    def __init__(self,data_list, student_path, teacher_path):
        self.file_names = data_list['audio_filename']
        self.labels = data_list.drop(columns=['split', 'audio_filename'])
        self.student_path = student_path
        self.teacher_path = teacher_path

    def __len__(self):    
        return len(self.file_names)

    def __getitem__(self,idx):
        sample = {}
        file = self.file_names.iloc[idx]
        
        student = np.load(self.student_path + '/' + file[:-4] + '.npy')
        student = fit_mel_size(student, mel_frames = 998)
        teacher = np.load(self.teacher_path + '/' + file[:-4] + '.npy')
        teacher = fit_mel_size(teacher, mel_frames = 998)
            
        student = np.expand_dims(student, axis=0)
        teacher = np.expand_dims(teacher, axis=0)
        target = self.labels.iloc[idx,:].to_numpy()
        
        student = torch.FloatTensor(student)
        teacher = torch.FloatTensor(teacher)
        target = torch.FloatTensor(target)
        
        sample['student'], sample['teacher'], sample['target'], sample['filename'] = student, teacher, target, file
        return sample


def train(student, teacher, T, train_load, c1, c2, c3, lr):

    optimizer = optim.Adam(student.parameters(),lr = lr)
    
    student.train()
    teacher.eval()

    for j, sample in enumerate(tqdm(train_load)):
        
        student_input = sample['student'].to(device)
        teacher_input = sample['teacher'].to(device)
        target = sample['target'].to(device) 
        student_input = student_input.float()
        teacher_input = teacher_input.float()
        target = target.float()
        optimizer.zero_grad()
       
        student_output, student_act = student(student_input)
        teacher_output, teacher_act = teacher(teacher_input)

        bce_loss = c1*bce_with_logits(student_output, target)
        kd_loss = c2*loss_fn_kd(student_output, teacher_output, T)
        sp_loss = c3*loss_fn_sp(student_act, teacher_act)
        loss = bce_loss + sp_loss + kd_loss

        loss.backward()
        optimizer.step()

def val(student, val_load, i):

    classes = ["1_engine", "2_machinery-impact", "3_non-machinery-impact", "4_powered-saw", "5_alert-signal", "6_music", "7_human-voice", "8_dog"]
    
    student.eval()
    predictions = pd.DataFrame(columns=["audio_filename", "1_engine", "2_machinery-impact", "3_non-machinery-impact", "4_powered-saw", "5_alert-signal", "6_music", "7_human-voice", "8_dog"])
    with torch.no_grad():
        for j, sample in enumerate(tqdm(val_load)):
            student_input = sample['student'].to(device)
            target = sample['target'].to(device) 
            filenames = sample['filename']
            student_input = student_input.float()
            target = target.float()
            output, _ = student(student_input)
            output = nn.Sigmoid()(output)
            for k in range(output.shape[0]):
                curr = output[k].detach().cpu().numpy()
                temp = {}
                temp["audio_filename"] = filenames[k]
                for p,class_name in enumerate(classes):
                    temp[class_name] = curr[p]
                predictions = predictions.append(temp, ignore_index=True)

    predictions.to_csv('pred/predictions_{}.csv'.format(i), index=False)
    df_dict = evaluate('pred/predictions_{}.csv'.format(i),'annotations-dev.csv','dcase-ust-taxonomy.yaml', "coarse")

    micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
    macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)
    return micro_auprc



