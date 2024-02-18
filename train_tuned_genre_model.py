# Standard python libs
import os
# Third-party libs
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
# PyTorch libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import pad
from torch.utils.data import DataLoader
# Local libs
from fmaDataset import fmaDataset
from genreModel import topGenreClassifier, train_model
from prepareDataset import buildDataframe

# This is our collate function, it allows the loader to process the features
# by making them the right "shape" for the model.
def resize_collate(batch):
    processed_features = [pad(features, (0, 2580 - features.shape[2]), "constant", 0) if features.shape[2] < 2580 else features[:, :, :2580] for features, label in batch]
    labels = torch.stack([torch.tensor(label) for _, label in batch])
    return torch.stack(processed_features), labels

# Loads a checkpoint for training
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

# This checks whether a file exists then writes appends to it.
def append_or_create(results_dataframe, file_path):
    if os.path.exists(file_path):
        existing_dataframe = pd.read_csv(file_path)
        updated_dataframe = pd.concat([existing_dataframe, results_dataframe], ignore_index=True)
    else:
        updated_dataframe = results_dataframe

    updated_dataframe.to_csv(file_path, index=False)

# This calculates our genre metrics and outputs them ready for file export
def calculate_metrics(metrics, labels_predictions, train_results, val_results, epoch):
    # Converting our labels to a format readable by sklearn etc.
    labels = torch.cat(labels_predictions['labels'], dim=0).numpy()
    predictions = torch.cat(labels_predictions['predictions'], dim=0).numpy()
    val_all_labels = np.concatenate(labels)
    val_all_predictions = np.concatenate(predictions)

    metrics['epoch'].append(epoch + 1)
    metrics['validation_accuracy'].append(sum(val_results['validation_accuracy']) / len(val_results['validation_accuracy']))
    metrics['train_accuracy'].append(sum(train_results['train_accuracy']) / len(train_results['train_accuracy']))
    metrics['validation_loss'].append(sum(val_results['validation_loss']) / len(val_results['validation_loss']))
    metrics['train_loss'].append(sum(train_results['train_loss']) / len(train_results['train_loss']))
    # Calculate our metrics based off labels and predictions
    metrics['f1_macro'].append(f1_score(val_all_labels, val_all_predictions, average='macro'))
    metrics['f1_micro'].append(f1_score(val_all_labels, val_all_predictions, average='micro'))
    metrics['hamming_loss'].append(hamming_loss(val_all_labels, val_all_predictions))
    metrics['subset_accuracy'].append(accuracy_score(val_all_labels, val_all_predictions))
    return metrics

def main():
    # If the data isnt ready i.e. you don't have the files, uncomment
    #prepareDataset.buildDataframe("genres", True)
    folder_path = 'track_genre_files'
    files = [file for file in os.listdir(folder_path) if file.endswith(".pkl")]

    # Setting our main hyperparameters
    batch_size = 25
    epoch_size = 25
    learning_rate = 0.0001

    # Filename for results
    results_file = str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-genre"+"_results.csv"

    # Initialise the model and find whether we have CUDA available
    # TODO, ensure non cuda support
    model = topGenreClassifier()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # If we need to use checkpoints e.g. process was stopped, uncomment
    #if os.path.exists("max_genre_v1.pth"):
        #model.load_state_dict(torch.load("max_genre_v1.pth"))

    #if os.path.exists("max_genre_checkpoint.pt"):
        #model, optimizer = load_ckp("max_genre_checkpoint.pt", model, optimizer)
    
    for epoch in range(epoch_size):
        # Set up all our storage variables
        metrics = {'epoch': [], 'f1_macro': [], 'f1_micro': [], 'hamming_loss': [], 'subset_accuracy': [], 'validation_accuracy': [], 'train_accuracy': [], 'validation_loss':[], 'train_loss':[]}
        epoch_val_results = {
            'validation_accuracy' : list(),
            'validation_loss' : list()
        }
        epoch_train_results = {
            'train_accuracy' : list(),
            'train_loss' : list()
        }
        epoch_labels_predictions = {
            'labels' : [],
            'predictions' : []
        }
        print("Epoch: "+str(epoch + 1)+"/"+str(epoch_size))
        for idx, file_name in enumerate(files):

            name, extension = os.path.splitext(file_name)
            if extension != ".pkl":
                continue
            
            # Read the file we are working on
            track_dataframe = pd.read_pickle(os.path.join(folder_path, file_name))
            track_dataframe.reset_index(drop=True, inplace=True)

            # Load the dataset and split into loaders
            dataset = fmaDataset(dataframe = track_dataframe, spectrogram = track_dataframe['spectrogram'].values, mfcc = track_dataframe['mfcc'], labels = track_dataframe['genre_vector'], id = track_dataframe['track_id'])
            train_df, test_df = train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=666)
            
            train_loader = DataLoader(train_df, batch_size = batch_size, collate_fn = resize_collate, shuffle=True)
            test_loader = DataLoader(test_df, batch_size = batch_size, collate_fn = resize_collate, shuffle=False)

            print("Analysing file: " + str(file_name) + "File no: "+ str(idx + 1) +"/"+ str(len(files)))
            print("Tracks to analyse: "+str(len(track_dataframe.index)+1))
            # Train the model with this file's tracks
            train_results, val_results, labels_predictions = train_model(model, train_loader, test_loader, criterion, optimizer, epoch, device)
            
            # There is some odd formatting with dictionaries, so this is here just to make sure
            # the format is right for metric calculations
            # TODO maybe look at re-doing the formats here and in the model code
            for label, prediction in zip(labels_predictions['labels'], labels_predictions['predictions']):
                epoch_labels_predictions['labels'].append(label)
                epoch_labels_predictions['predictions'].append(prediction)

            epoch_val_results['validation_accuracy'].append(val_results['val_accuracy'])
            epoch_train_results['train_accuracy'].append(train_results['train_accuracy'])

            epoch_train_results['train_loss'].append(train_results['train_loss'])
            epoch_val_results['validation_loss'].append(val_results['val_loss'])
        
        metrics = calculate_metrics(metrics, epoch_labels_predictions, epoch_train_results, epoch_val_results, epoch)
        append_or_create(pd.DataFrame(metrics), results_file)

        # Save our progress just in case!
        torch.save({
            'epoch' : epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, "max_genre_checkpoint.pt")

    torch.save(model.state_dict(), "max_genre_v3.pth")

if __name__ == "__main__":
    main()