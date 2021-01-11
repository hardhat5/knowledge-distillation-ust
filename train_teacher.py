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
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader    
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc
from tqdm import tqdm, trange
from utils import fit_mel_size
from models import DCASE_Net, MobileNetV2
from loss_fn import bce_with_logits
from configparser import ConfigParser

parser = ConfigParser()
parser.read('config.ini')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_list = pd.read_csv('train.csv')
val_list = pd.read_csv('validate.csv')

classes = ["1_engine", "2_machinery-impact", "3_non-machinery-impact", "4_powered-saw", "5_alert-signal", "6_music", "7_human-voice", "8_dog"]

class AudioDataset(Dataset):

    def __init__(self,data_list, data_path):
        self.file_names = data_list['audio_filename']
        self.labels = data_list.drop(columns=['split', 'audio_filename'])
        self.path = data_path

    def __len__(self):    
        return len(self.file_names)

    def __getitem__(self,idx):
        sample = {}
        file = self.file_names.iloc[idx]
        data = np.load(self.path + '/' + file[:-4] + '.npy')
        data = fit_mel_size(data, mel_frames = 998)
            
        data = np.expand_dims(data, axis=0)
        target = self.labels.iloc[idx,:].to_numpy()
        
        data = torch.FloatTensor(data)
        target = torch.FloatTensor(target)
        
        sample['data'], sample['target'], sample['filename'] = data, target, file
        return sample

train_dataset = AudioDataset(train_list, 'features/teacher/')
train_load = DataLoader(train_dataset, batch_size = 8, shuffle=True, num_workers = 0)

val_dataset = AudioDataset(val_list, 'features/teacher/')
val_load = DataLoader(val_dataset, batch_size = 8, shuffle=True, num_workers = 0)

if(parser.get('teacher', 'MODEL')=='dcase_net'):
    teacher_model = DCASE_Net()
    teacher_model = teacher_model.cuda()

    state_dict = torch.load('vggish_pretrained_convs.pth', map_location=device)
    own_state = teacher_model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        param = param.data
        own_state[name].copy_(param)

    names = ['layer1_conv1.0.weight', 'layer1_conv1.0.bias', 'layer3_conv2.0.weight', 'layer3_conv2.0.bias', 'layer5_conv3_1.0.weight', 'layer5_conv3_1.0.bias', 'layer6_conv3_2.0.weight', 'layer6_conv3_2.0.bias']
    for name, param in teacher_model.named_parameters():
        if name in names:
            param.requires_grad = False

else:
    width_mult = parser.getfloat('teacher', 'WIDTH_MULT')
    teacher_model = MobileNetV2(width_mult, 8)
    teacher_model = teacher_model.cuda()

epochs = parser.getint('teacher', 'EPOCHS')

lr = parser.getfloat('teacher', 'LEARNING_RATE')
criterion_teacher = nn.BCELoss()
optimiser_teacher = optim.Adam(teacher_model.parameters(),lr = lr)

max_acc = 0.0

for i in range(epochs):
    
    predictions = pd.DataFrame(columns=["audio_filename", "1_engine", "2_machinery-impact", "3_non-machinery-impact", "4_powered-saw", "5_alert-signal", "6_music", "7_human-voice", "8_dog"])
    teacher_model.train()
    
    print("Training epoch {}".format(i))
    for j, sample in enumerate(tqdm(train_load)):
        data = sample['data'].to(device)
        target = sample['target'].to(device) 
        data = data.float()
        target = target.float()
        optimiser_teacher.zero_grad()
        output, _ = teacher_model(data)
        loss = bce_with_logits(output, target)
        loss.backward()
        optimiser_teacher.step()
    
    teacher_model.eval()
    predictions = pd.DataFrame(columns=["audio_filename", "1_engine", "2_machinery-impact", "3_non-machinery-impact", "4_powered-saw", "5_alert-signal", "6_music", "7_human-voice", "8_dog"])
    with torch.no_grad():
        for j, sample in enumerate(tqdm(val_load)):
            student_input = sample['data'].to(device)
            target = sample['target'].to(device) 
            filenames = sample['filename']
            student_input = student_input.float()
            target = target.float()
            output, _ = teacher_model(student_input)
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

    print("Micro AUPRC: ", micro_auprc)
        
    if micro_auprc>max_acc:
        torch.save(teacher_model.state_dict(), 'weights/teacher')
        max_acc = micro_auprc

print("Maximum micro AUPRC: ", max_acc)

