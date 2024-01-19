from featureModel import audioFeatureModel
from featureModel import train_model
import prepareDataset
from fmaDataset import fmaDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from featureModel import audioFeatureModel
from featureModel import train_model
import torch.cuda
from torch.nn.functional import pad
import os
import pandas as pd
import pickle

def normalize_tempo(split):
    tempo = [item[1][-1] for item in split]
    min_tempo = min(tempo)
    max_tempo = max(tempo)
    normalized_data = []
    for tensor_data, scalar_values in split:
        normalized_last_value = (scalar_values[-1] - min_tempo) / (max_tempo - min_tempo)
        
        normalized_scalar_values = scalar_values[:-1] + (normalized_last_value,)

        normalized_data.append((tensor_data, normalized_scalar_values))
    
    return(normalized_data, min_tempo, max_tempo)

def resize_collate(batch):
    processed_features = []
    labels = []

    for features, label in batch:
        current_size = features.shape[2]
        if current_size < 2580:
            padding_needed = 2580 - current_size
            features = pad(features, (0, padding_needed), "constant", 0)
        elif current_size > 2580:
            features = features[:, :, :2580]
        processed_features.append(features)

        labels.append(torch.tensor(label))

    labels = torch.stack(labels)
    features_batch = torch.stack(processed_features)

    return features_batch, labels

if not os.path.exists("tracks_features_1000.pkl"):
    track_dataframe = prepareDataset.buildDataframe("features")
    with open('tracks_features_1000.pkl', 'wb') as file:
        pickle.dump(track_dataframe, file)
else:
    track_dataframe = pd.read_pickle('tracks_features_1000.pkl')

print("Tracks to analyse: "+str(len(track_dataframe.index)+1))

rows_with_na = track_dataframe.isna().any(axis=0)
na_rows_exist = rows_with_na.any()
if na_rows_exist:
    rows_with_na_df = track_dataframe[rows_with_na]
    print(str(rows_with_na_df)+" rows with missing data")

batch_size = 12
epoch_size = 500
learning_rate = 0.0001

track_dataframe.reset_index(drop=True, inplace=True)

dataset = fmaDataset(dataframe = track_dataframe, id = track_dataframe['track_id'], spectrogram = track_dataframe['spectrogram'].values, mfcc = track_dataframe['mfcc'], labels = track_dataframe['features'])

train_df, test_df = train_test_split(dataset, test_size=0.3, train_size=0.7, random_state=666)

train_df, train_min, train_max = normalize_tempo(train_df)
test_df, val_min, val_max = normalize_tempo(test_df)

normalization_values = {
    "train_max" : train_max,
    "train_min" : train_min,
    "val_max" : val_max,
    "val_min" : val_min
}

train_loader = DataLoader(train_df, batch_size = batch_size, collate_fn = resize_collate, shuffle=True)
test_loader = DataLoader(test_df, batch_size = batch_size, collate_fn = resize_collate, shuffle=False)

model = audioFeatureModel()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

train_results, val_results = train_model(model, train_loader, test_loader, criterion, optimizer, epoch_size, device, normalization_values)

train_results_dataframe = pd.DataFrame(train_results)
val_results_dataframe = pd.DataFrame(val_results)
train_results_dataframe.to_csv(str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-"+'train_results.csv')
val_results_dataframe.to_csv(str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-"+'val_results.csv')