# Third-party libs
import mysql.connector
import pandas as pd
# PyTorch libs
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# Local libs
from databaseConfig import connect

db = connect()

class fmaDataset(Dataset):
    def __init__(self, dataframe,beats, spectrogram, mfcc, labels, id):
        self.dataframe = dataframe
        self.beats = beats
        self.spectrogram = spectrogram
        self.mfcc = mfcc
        self.id = id
        self.labels = labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        spec = self.spectrogram[idx]
        mfcc = self.mfcc[idx]
        beats = self.beats[idx]

        spec = spec[..., :2580]
        mfcc = mfcc[..., :2580]

        if mfcc.shape[0] == 1:
            mfcc = mfcc.repeat(2, 1, 1)
        if spec.shape[0] == 1:
            spec = spec.repeat(2, 1, 1)

        valid_beats = [beat for beat in beats[1] if beat < mfcc.shape[2]]
        beat_vector = torch.zeros(mfcc.shape[2])
        if valid_beats:
            valid_beats_tensor = torch.tensor(valid_beats, dtype=torch.long)
            beat_vector[valid_beats_tensor] = 1
        beat_vector = beat_vector.unsqueeze(0).unsqueeze(0)
        beat_vector = beat_vector.repeat(2, 1, 1)
        
        combined_features = torch.cat([spec, mfcc, beat_vector], dim=1)
    
        label = self.labels[idx]

        return combined_features, label

