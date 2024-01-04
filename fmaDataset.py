import torch
from torch.utils.data import Dataset, DataLoader
import mysql.connector
from databaseConfig import connect
import pandas as pd

db = connect()

class fmaDataset(Dataset):
    def __init__(self, dataframe, spectrogram, centroid, labels):
        self.dataframe = dataframe
        self.spectrogram = spectrogram
        self.centroid = centroid
        self.label_col = labels 
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get Mel spectrogram and spectral centroid for the current index
        spec = self.mel_spectrograms[idx]
        centroid = self.spectral_centroids[idx]

        # Reshape or expand the spectral centroid to match the Mel spectrogram dimensions
        spec_centroid_expanded = centroid.unsqueeze(0).expand(spec.size(0), -1)

        # Stack along a new dimension to create multiple channels
        combined_features = torch.stack([spec, spec_centroid_expanded], dim=0)

        # Retrieve the label
        label = self.labels[idx]

        return combined_features, label

