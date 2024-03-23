# Standard Python libs
import os
import numpy as np
import pandas as pd
# Third-party libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
# PyTorch libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.cuda
# Local libs
from max_utils import append_or_create, load_ckp
from featureModel import audioFeatureModel, train_model
import prepareDataset
from fmaDataset import fmaDataset

# Here we are uisng min-max normalization
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


def initialize_metrics():
    return {'epoch': [], 'validation_loss':[], 'train_loss':[], 'pearsonr':[], 'r2_score':[]}


# This is our collate function, it allows the loader to process the features
# by making them the right "shape" for the model.
def resize_collate(batch):
    processed_features = [pad(features, (0, max(0, 2580 - features.shape[2])), "constant", 0) if features.shape[2] < 2580 else features[:, :, :2580] for features, _ in batch]
    labels = [torch.tensor(label) for _, label in batch]
    return torch.stack(processed_features), torch.stack(labels)

# This calculates our genre metrics and outputs them ready for file export
def calculate_metrics(metrics, labels_predictions, train_results, val_results, epoch):
    labels = torch.cat(labels_predictions['labels'], dim=0).numpy()
    predictions = torch.cat(labels_predictions['predictions'], dim=0).numpy()
    val_all_labels = np.concatenate(labels)
    val_all_predictions = np.concatenate(predictions)
    metrics['epoch'].append(epoch + 1)
    metrics['validation_loss'].append(sum(val_results['validation_loss']) / len(val_results['validation_loss']))
    metrics['train_loss'].append(sum(train_results['train_loss']) / len(train_results['train_loss']))
    # Calculate our metrics based off labels and predictions
    metrics['pearsonr'].append(pearsonr(val_all_labels, val_all_predictions)[0])
    metrics['r2_score'].append(r2_score(val_all_labels, val_all_predictions))
    return metrics

def main():
    # If the data isnt ready i.e. you don't have the files, uncomment
    #prepareDataset.buildDataframe("features", True)
    folder_path = 'feature_track_files'
    files = [file for file in os.listdir(folder_path) if file.endswith(".pkl")]

    # Setting our main hyperparameters
    batch_sizes = [10,25,50]
    learning_rates = [0.0001,0.001,0.01,0.1,0.25]
    epoch_sizes = [10,25,50,100,250,500]

    for epoch_size in epoch_sizes:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                # Filename for results
                results_file = str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-feature"+"_results.csv"
                print("epoch_size : "+str(epoch_size)+"-batch_size: "+str(batch_size)+"-learning_rate: "+str(learning_rate))
                # Initialise the model and find whether we have CUDA available
                # TODO, ensure non cuda support
                model = audioFeatureModel()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()
                scaler = GradScaler()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                # If we need to use checkpoints e.g. process was stopped, uncomment
                #if os.path.exists("max_feature_v1.pth"):
                    #model.load_state_dict(torch.load("max_feature_v1.pth"))

                #if os.path.exists("max_feature_checkpoint.pt"):
                    #model, optimizer = load_ckp("max_feature_checkpoint.pt", model, optimizer)
                
                for epoch in range(epoch_size):
                    # Set up all our storage variables
                    metrics = initialize_metrics()

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
                        dataset = fmaDataset(dataframe=track_dataframe, id=track_dataframe['track_id'], spectrogram=track_dataframe['spectrogram'].values, mfcc=track_dataframe['mfcc'], beats=track_dataframe['beats'], labels=track_dataframe['features'])
                        train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=666)
                        # Normalize the tempo column of our data
                        train_df, train_min, train_max = normalize_tempo(train_df)
                        test_df, val_min, val_max = normalize_tempo(test_df)

                        train_loader = DataLoader(train_df, batch_size=batch_size, collate_fn=resize_collate, shuffle=True)
                        test_loader = DataLoader(test_df, batch_size=batch_size, collate_fn=resize_collate, shuffle=False)
                        
                        print("Analysing file: " + str(file_name) + "File no: "+ str(idx + 1) +"/"+ str(len(files)))
                        print("Tracks to analyse: "+str(len(track_dataframe.index)+1))
                        # Train the model with this file's tracks
                        train_results, val_results, labels_predictions = train_model(model, train_loader, test_loader, criterion, optimizer, scaler, epoch, device)
                        
                        # There is some odd formatting with dictionaries, so this is here just to make sure
                        # the format is right for metric calculations
                        # TODO maybe look at re-doing the formats here and in the model code
                        for label, prediction in zip(labels_predictions['labels'], labels_predictions['predictions']):
                            epoch_labels_predictions['labels'].append(label)
                            epoch_labels_predictions['predictions'].append(prediction)

                        epoch_train_results['train_loss'].append(train_results['train_loss'])
                        epoch_val_results['validation_loss'].append(val_results['val_loss'])
                    
                    metrics = calculate_metrics(metrics, epoch_labels_predictions, epoch_train_results, epoch_val_results, epoch)
                    append_or_create(pd.DataFrame(metrics), results_file)

                    # Save our progress just in case!
                    torch.save({
                        'epoch' : epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, "max_feature_checkpoint.pt")

            torch.save(model.state_dict(), "max_feature_v3.pth")

if __name__ == "__main__":
    main()