import prepareDataset
from fmaDataset import fmaDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from genreModel import topGenreClassifier
from genreModel import train_model
import torch.cuda
from torch.nn.functional import pad
import os
import pandas as pd
import pickle


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
        elif features.shape[0] == 1:
            features = features.repeat(2, 1, 1)
        processed_features.append(features)

        labels.append(torch.tensor(label))

    labels = torch.stack(labels)
    features_batch = torch.stack(processed_features)

    return features_batch, labels

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

folder_path = 'track_genre_files'
files = os.listdir(folder_path)

for idx, file_name in enumerate(files):
    name, extension = os.path.splitext(file_name)

    if extension != ".pkl":
        continue

    track_dataframe = pd.read_pickle(os.path.join(folder_path, file_name))

    print("Analysing file: " + str(file_name) + "File no: "+ str(idx) +"/"+ str(len(files)))
    print("Tracks to analyse: "+str(len(track_dataframe.index)+1))

    rows_with_na = track_dataframe.isna().any(axis=0)
    na_rows_exist = rows_with_na.any()
    if na_rows_exist:
        rows_with_na_df = track_dataframe[rows_with_na]
        print(str(rows_with_na_df)+" rows with missing data")

    batch_size = 25
    epoch_size = 2
    learning_rate = 0.0001

    track_dataframe.reset_index(drop=True, inplace=True)

    dataset = fmaDataset(dataframe = track_dataframe, spectrogram = track_dataframe['spectrogram'].values, mfcc = track_dataframe['mfcc'], labels = track_dataframe['genre_vector'], id = track_dataframe['track_id'])

    train_df, test_df = train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=666)

    batch_size = batch_size
    train_loader = DataLoader(train_df, batch_size = batch_size, collate_fn = resize_collate, shuffle=True)
    test_loader = DataLoader(test_df, batch_size = batch_size, collate_fn = resize_collate, shuffle=False)

    model = topGenreClassifier()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if idx > 0:
        if os.path.exists("max_genre_checkpoint.pt"):
            model, optimizer = load_ckp("max_genre_checkpoint.pt", model, optimizer)

    train_results, val_results = train_model(model, train_loader, test_loader, criterion, optimizer, epoch_size, device)

    train_results_dataframe = pd.DataFrame(train_results)
    val_results_dataframe = pd.DataFrame(val_results)
    train_results_dataframe.to_csv(str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-"+str(idx)+"-train_results.csv")
    val_results_dataframe.to_csv(str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-"+str(idx)+"val_results.csv")
    torch.save(model.state_dict(), "max_genre_v1.pth")