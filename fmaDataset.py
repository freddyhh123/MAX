import torch
from torch.utils.data import Dataset, DataLoader
import mysql.connector
from databaseConfig import connect
import pandas as pd
import torch.nn.functional as F

db = connect()

class fmaDataset(Dataset):
    def __init__(self, dataframe, spectrogram, mfcc, labels):
        self.dataframe = dataframe
        self.spectrogram = spectrogram
        self.mfcc = mfcc
        self.labels = labels 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        spec = self.spectrogram[idx]
        mfcc = self.mfcc[idx]
        spec = spec[..., :2580]
        mfcc = mfcc[..., :2580]

        combined_features = torch.cat([spec, mfcc], dim=1)
    
        label = self.labels[idx]

        return combined_features, label

