import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import pad
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split

from prepareDataset import buildDataframe
from fmaDataset import fmaDataset
from genreModel import topGenreClassifier, train_model


def resize_collate(batch):
    processed_features = [pad(features, (0, 2580 - features.shape[2]), "constant", 0) if features.shape[2] < 2580 else features[:, :, :2580] for features, label in batch]
    labels = torch.stack([torch.tensor(label) for _, label in batch])
    return torch.stack(processed_features), labels

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

def calculate_metrics(metrics, labels_predictions, train_results, val_results, epoch):
    labels = torch.cat(labels_predictions['labels'], dim=0).numpy()
    predictions = torch.cat(labels_predictions['predictions'], dim=0).numpy()
    val_all_labels = np.concatenate(labels)
    val_all_predictions = np.concatenate(predictions)
    metrics['epoch'].append(epoch + 1)
    metrics['validation_accuracy'].append(val_results['val_accuracy'])
    metrics['train_accuracy'].append(train_results['train_accuracy'])
    metrics['f1_macro'].append(f1_score(val_all_labels, val_all_predictions, average='macro'))
    metrics['f1_micro'].append(f1_score(val_all_labels, val_all_predictions, average='micro'))
    metrics['hamming_loss'].append(hamming_loss(val_all_labels, val_all_predictions))
    metrics['subset_accuracy'].append(accuracy_score(val_all_labels, val_all_predictions))
    return metrics

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

    #if os.path.exists("max_genre_v1.pth"):
        #model.load_state_dict(torch.load("max_genre_v1.pth"))

    #if os.path.exists("max_genre_checkpoint.pt"):
        #model, optimizer = load_ckp("max_genre_checkpoint.pt", model, optimizer)

    epoch_validation_accuracy = list()
    epoch_train_accuracy = list()
    for epoch in range(epoch_size):
        print("Epoch: "+str(epoch)+"/"+str(epoch_size))
        for idx, file_name in enumerate(files):
            metrics = {'epoch': [], 'f1_macro': [], 'f1_micro': [], 'hamming_loss': [], 'subset_accuracy': [], 'validation_accuracy': [], 'train_accuracy': []}
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
            
            epoch_validation_accuracy.append(val_results['val_accuracy'])
            epoch_train_accuracy.append(train_results['train_accuracy'])
        
            metrics = calculate_metrics(metrics, labels_predictions, train_results, val_results, epoch)
            append_or_create(pd.DataFrame(metrics), val_results_file)

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, "max_genre_checkpoint.pt")

    torch.save(model.state_dict(), "max_genre_v2.pth")

if __name__ == "__main__":
    main()