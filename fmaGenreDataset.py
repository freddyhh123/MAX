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
    """
    A custom Dataset class for handling the FMA dataset. Including sub genres.

    Attributes:
    - dataframe (pd.DataFrame): DataFrame containing metadata or other information related to the tracks.
    - beats (list): A list where each element corresponds to beat locations in the tracks.
    - spectrogram (list): A list of spectrogram tensors for the tracks.
    - mfcc (list): A list of MFCC tensors for the tracks.
    - id (list): A list of identifiers for each track.
    - labels (list): A list of labels representing the top genre for each track.
    - sub_genre_labels (list): A list of labels representing the sub-genre for each track.

    Parameters:
    - dataframe (pd.DataFrame): Data related to the tracks.
    - beats (list): Beat information for each track.
    - spectrogram (list): Spectrograms of the tracks.
    - mfcc (list): MFCC features of the tracks.
    - top_genres (list): Top genre labels for each track.
    - sub_genres (list): Sub-genre labels for each track.
    - id (list): Identifiers for each track.
    """
    def __init__(self, dataframe, beats , spectrogram, mfcc, top_genres, sub_genres, id):
        self.dataframe = dataframe
        self.beats = beats
        self.spectrogram = spectrogram
        self.mfcc = mfcc
        self.id = id
        self.labels = top_genres
        self.sub_genre_labels = sub_genres

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        - int: The total number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Parameters:
        - idx (int): Index of the data item to retrieve.

        Returns:
        - tuple: A tuple containing combined features (torch.Tensor) and the corresponding label.
        """
        spec = self.spectrogram[idx]
        mfcc = self.mfcc[idx]
        beats = self.beats[idx]

        # Ensure uniform dimensions
        spec = spec[..., :2580]
        mfcc = mfcc[..., :2580]

        # Duplicate channel if necessary
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
        sub_genre_labels = self.sub_genre_labels[idx]

        return combined_features, label, sub_genre_labels

