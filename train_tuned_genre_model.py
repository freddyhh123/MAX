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
from sklearn.metrics import f1_score, accuracy_score, hamming_loss


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

def append_or_create(results_dataframe, file_path):
    if os.path.exists(file_path):
        existing_dataframe = pd.read_csv(file_path)
        updated_dataframe = pd.concat([existing_dataframe, results_dataframe], ignore_index=True)
    else:
        updated_dataframe = results_dataframe

    updated_dataframe.to_csv(file_path, index=False)

def main():
    #prepareDataset.buildDataframe("genres", True)
    folder_path = 'track_genre_files'
    files = [file for file in os.listdir(folder_path) if file.endswith(".pkl")]

    batch_size = 25
    epoch_size = 25
    learning_rate = 0.0001

    train_results_file = str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-genre"+"-train_results.csv"
    val_results_file = str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-genre"+"-val_results.csv"

    model = topGenreClassifier()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if os.path.exists("max_genre_v1.pth"):
        model.load_state_dict(torch.load("max_genre_v1.pth"))

    if os.path.exists("max_genre_checkpoint.pt"):
        model, optimizer = load_ckp("max_genre_checkpoint.pt", model, optimizer)

    f1_macro_scores = list()
    f1_micro_scores = list()
    hammingloss = list()
    subset_accuracy = list()
    for epoch in range(epoch_size):
        label_predictions_epoch = list()
        for idx, file_name in enumerate(files):
            name, extension = os.path.splitext(file_name)
            if extension != ".pkl":
                continue
            track_dataframe = pd.read_pickle(os.path.join(folder_path, file_name))
            track_dataframe.reset_index(drop=True, inplace=True)
            rows_with_na = track_dataframe.isna().any(axis=0)
            na_rows_exist = rows_with_na.any()
            if na_rows_exist:
                rows_with_na_df = track_dataframe[rows_with_na]
                print(str(rows_with_na_df)+" rows with missing data")

            dataset = fmaDataset(dataframe = track_dataframe, spectrogram = track_dataframe['spectrogram'].values, mfcc = track_dataframe['mfcc'], labels = track_dataframe['genre_vector'], id = track_dataframe['track_id'])
            train_df, test_df = train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=666)
            
            train_loader = DataLoader(train_df, batch_size = batch_size, collate_fn = resize_collate, shuffle=True)
            test_loader = DataLoader(test_df, batch_size = batch_size, collate_fn = resize_collate, shuffle=False)

            print("Analysing file: " + str(file_name) + "File no: "+ str(idx) +"/"+ str(len(files)))
            print("Tracks to analyse: "+str(len(track_dataframe.index)+1))
            train_results, val_results, labels_predictions = train_model(model, train_loader, test_loader, criterion, optimizer, epoch, device)
            
            label_predictions_epoch.append(labels_predictions)

            train_results_dataframe = pd.DataFrame(train_results)
            val_results_dataframe = pd.DataFrame(val_results)
        print(label_predictions_epoch)
        
        f1_macro_scores.append(f1_score(labels_predictions['labels'], labels_predictions['predictions'], average='macro'))
        f1_micro_scores.append(f1_score(labels_predictions['labels'], labels_predictions['predictions'], average='micro'))
        hammingloss.append(hamming_loss(labels_predictions['labels'], labels_predictions['predictions']))
        subset_accuracy.append(accuracy_score(labels_predictions['labels'], labels_predictions['predictions']))
        print(f1_macro_scores, f1_micro_scores, hammingloss, subset_accuracy)

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, "max_genre_checkpoint.pt")

    torch.save(model.state_dict(), "max_genre_v2.pth")

if __name__ == "__main__":
    main()