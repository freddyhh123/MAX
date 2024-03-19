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

class fmaGenreDataset(Dataset):
    def __init__(self, dataframe, beats , spectrogram, mfcc, top_genres, sub_genres, id):
        self.dataframe = dataframe
        self.beats = beats
        self.spectrogram = spectrogram
        self.mfcc = mfcc
        self.id = id
        self.labels = top_genres
        self.sub_genre_labels = sub_genres

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

        beat_vector = torch.zeros(mfcc.shape[2])
        beat_vector[beats[1]] = 1
        beat_vector = beat_vector.unsqueeze(0).unsqueeze(0)
        beat_vector = beat_vector.repeat(2,1,1)

        combined_features = torch.cat([spec, mfcc, beat_vector], dim=1)
    
        label = self.labels[idx]
        sub_genre_labels = self.sub_genre_labels[idx]

        return combined_features, label, sub_genre_labels

