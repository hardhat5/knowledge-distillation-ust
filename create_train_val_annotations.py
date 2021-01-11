# Code to create csv files containing labels for the different audio files

import numpy as np
import pandas as pd

df = pd.read_csv("annotations-dev.csv", low_memory=False)
retain = [0, 2, 3, -8, -7, -6, -5, -4, -3, -2, -1]
df = df.iloc[:,retain]
cols = df.columns.to_list()

valid = df[df['split'] == "validate"]
valid = valid[valid['annotator_id'] == 0]
valid = valid[cols]
valid = valid.drop(columns=["annotator_id"])
valid.reset_index()
valid.to_csv("validate.csv", index=False)

train_files = df[df['split'] == "train"]
files = train_files['audio_filename'].unique()
train = pd.DataFrame()

cat = df.columns[-8:]
for file in files:
    temp = df[df['audio_filename']==file]
    d = temp.iloc[0,:].to_dict()
    for index, row in temp.iterrows():
        for c in cat:
            if(d[c]==0): d[c] = row[c]
                
    train = train.append(d, ignore_index=True) 

train = train[cols]
train = train.drop(columns=["annotator_id"])
train.to_csv("train.csv", index=False)