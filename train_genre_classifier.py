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
from torch.cuda.amp import GradScaler
# Local libs
from maxUtils import load_ckp, append_or_create
from fmaGenreDataset import fmaGenreDataset
from genreModel import topGenreClassifier, train_model
from subGenreModel import SubGenreClassifier, train_sub_models
from prepareDataset import buildDataframe

# This is our collate function, it allows the loader to process the features
# by making them the right "shape" for the model.
def resize_collate(batch):
    processed_features = [pad(features, (0, 2580 - features.shape[2]), "constant", 0) if features.shape[2] < 2580 else features[:, :, :2580] for features, label, sub_genre_label in batch]
    genre_info = [{'top_genre': top_genre, 'sub_genre': sub_genre} 
                  for _, top_genre, sub_genre in batch]
    return torch.stack(processed_features), genre_info

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

def initialize_metrics(sub_genre_models):
    for idx in sub_genre_models:
        sub_genre_models[idx]['val_loss'] = 0.0
        sub_genre_models[idx]['train_loss'] = 0.0
        sub_genre_models[idx]['val_accuracy'] = []
        sub_genre_models[idx]['train_accuracy'] = []
        sub_genre_models[idx]['f1_micro'] = []
        sub_genre_models[idx]['train_count'] = 0
        sub_genre_models[idx]['val_count'] = 0
    return {'epoch': [], 'f1_macro': [], 'f1_micro': [], 'hamming_loss': [], 'subset_accuracy': [], 'validation_accuracy': [], 'train_accuracy': [], 'validation_loss':[], 'train_loss':[]}, sub_genre_models

def update_epoch_results(epoch_results, train_results, val_results):
    epoch_results['validation_accuracy'].append(val_results['val_accuracy'])
    epoch_results['train_accuracy'].append(train_results['train_accuracy'])
    epoch_results['train_loss'].append(train_results['train_loss'])
    epoch_results['validation_loss'].append(val_results['val_loss'])
    return epoch_results

def average_sub_genre_metrics(sub_genre_models, epoch):
    metrics_to_keep = ['val_loss', 'train_loss', 'val_accuracy', 'train_accuracy', 'f1_micro','train_count','val_count']
    metrics_to_average = ['val_loss','train_loss','val_accuracy', 'train_accuracy', 'f1_micro']
    sub_genre_metrics = []

    for model_id, metrics in sub_genre_models.items():
        record = {"model_id": model_id, "epoch": epoch}
        
        for metric in metrics_to_keep:
            if metric in metrics:
                value = metrics[metric]
                # Check if this metric needs averaging
                if isinstance(value, list) and metric in metrics_to_average:
                    record[metric] = sum(value) / len(value) if value else None
                elif metric in metrics_to_average:
                    if metric == 'val_loss':
                        record[metric] = value / metrics['val_count'] if value else None
                    elif metric == 'train_loss':
                        record[metric] = value / metrics['train_count'] if value else None
                else:
                    record[metric] = value
            else:
                record[metric] = None
        
        sub_genre_metrics.append(record)

    # Convert to dataframe for saving
    return(pd.DataFrame(sub_genre_metrics))

def save_sub_models(sub_genre_models):
    for idx, model_info in sub_genre_models.items():
        model = model_info['model']
        file_name = "max_sub_genre_"+str(idx)+".pth"
        torch.save(model.state_dict(), file_name)

def initialize_sub_genre_models(learning_rate):
    # info for generating sub-genre models, left side is genre_id
    # right side is the number of sub_genres or outputs
    sub_genre_models = {}
    sub_genre_info = [30,2,7,8,1,6,3,29,4,4,19,6,12,8,19,5]
    for idx, sub_genres in enumerate(sub_genre_info):
                    sub_model = SubGenreClassifier(num_classes = sub_genres)
                    sub_genre_models[idx] = {
                        'model' : sub_model,
                        'optomizer': optim.Adam(sub_model.parameters(), lr=learning_rate),
                        'val_loss' : None,
                        'train_loss': None,
                        'val_accuracy' : [],
                        'train_accuracy' : [],
                        'f1_micro': [],
                        'val_count' : None,
                        'train_count': None
                    }
    return sub_genre_models

def main():
    # If the data isnt ready i.e. you don't have the files, uncomment
    #buildDataframe("genres", True)
    folder_path = 'genre_beats'
    files = [file for file in os.listdir(folder_path) if file.endswith(".pkl")]

    # Setting our main hyperparameters
    batch_sizes = [10,25, 50]
    learning_rates = [0.0001,0.001,0.01,0.1,0.25]
    epoch_sizes = [10,25,50,100,250]

    for epoch_size in epoch_sizes:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                # Filename for results
                results_file = str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-genre"+"_results.csv"
                sub_genre_results_file = str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-sub_genre"+"_results.csv"

                print("epoch_size : "+str(epoch_size)+" - batch_size: "+str(batch_size)+" - learning_rate: "+str(learning_rate))
                # Initialise the model and find whether we have CUDA available
                # TODO, ensure non cuda support
                model = topGenreClassifier()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.BCEWithLogitsLoss()
                top_genre_scaler = GradScaler()
                sub_genre_scaler = GradScaler()
                sub_genre_models = initialize_sub_genre_models(learning_rate)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                    
                # If we need to use checkpoints e.g. process was stopped, uncomment
                #if os.path.exists("max_genre_v1.pth"):
                    #model.load_state_dict(torch.load("max_genre_v1.pth"))

                #if os.path.exists("max_genre_checkpoint.pt"):
                    #model, optimizer = load_ckp("max_genre_checkpoint.pt", model, optimizer)
                
                for epoch in range(epoch_size):
                    # Set up all our storage variables
                    metrics, sub_genre_models = initialize_metrics(sub_genre_models)
            
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
                        dataset = fmaGenreDataset(dataframe = track_dataframe, spectrogram = track_dataframe['spectrogram'].values, mfcc = track_dataframe['mfcc'],beats = track_dataframe['beats'], top_genres = track_dataframe['genre_vector'], sub_genres = track_dataframe['sub_genre_vectors'], id = track_dataframe['track_id'])
                        train_df, test_df = train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=666)
                        
                        train_loader = DataLoader(train_df, batch_size = batch_size, collate_fn = resize_collate, shuffle=True)
                        test_loader = DataLoader(test_df, batch_size = batch_size, collate_fn = resize_collate, shuffle=False)

                        print("Analysing file: " + str(file_name) + "File no: "+ str(idx + 1) +"/"+ str(len(files)))
                        print("Tracks to analyse: "+str(len(track_dataframe.index)+1))
                        # Train the model with this file's tracks
                        train_results, val_results, labels_predictions = train_model(model, train_loader, test_loader, criterion, optimizer, top_genre_scaler, epoch, device)
                        sub_genre_models = train_sub_models(sub_genre_models, train_loader, test_loader, criterion, sub_genre_scaler, epoch, device)
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
                    append_or_create(average_sub_genre_metrics(sub_genre_models, epoch), sub_genre_results_file)

                    # Save our progress just in case!
                    torch.save({
                        'epoch' : epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, "max_genre_checkpoint.pt")

            torch.save(model.state_dict(), "max_genre_v3.pth")
            save_sub_models(sub_genre_models)

if __name__ == "__main__":
    main()