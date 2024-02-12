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
    prepareDataset.buildDataframe("features", True)
    folder_path = 'track_files'
    files = [file for file in os.listdir(folder_path) if file.endswith(".pkl")]

    count = 0
    batch_size = 25
    epoch_size = 50
    learning_rate = 0.0001

    train_results_file = str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-feature"+"-train_results.csv"
    val_results_file = str(epoch_size)+"-"+str(batch_size)+"-"+str(learning_rate)+"-feature"+"-val_results.csv"

    model = audioFeatureModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if os.path.exists("max_feature_v1.pth"):
        model.load_state_dict(torch.load("max_feature_v1.pth"))

    if os.path.exists("max_feature_checkpoint.pt"):
        model, optimizer = load_ckp("max_feature_checkpoint.pt", model, optimizer)

    for epoch in range(epoch_size):
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

            dataset = fmaDataset(dataframe=track_dataframe, id=track_dataframe['track_id'], spectrogram=track_dataframe['spectrogram'].values, mfcc=track_dataframe['mfcc'], labels=track_dataframe['features'])
            train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=666)
            train_df, train_min, train_max = normalize_tempo(train_df)
            test_df, val_min, val_max = normalize_tempo(test_df)

            train_loader = DataLoader(train_df, batch_size=batch_size, collate_fn=resize_collate, shuffle=True)
            test_loader = DataLoader(test_df, batch_size=batch_size, collate_fn=resize_collate, shuffle=False)
            
            print("Analysing file: " + str(file_name) + "File no: "+ str(idx) +"/"+ str(len(files)))
            print("Tracks to analyse: "+str(len(track_dataframe.index)+1))
            train_results, val_results = train_model(model, train_loader, test_loader, criterion, optimizer, epoch, device, {'train_max': train_max, 'train_min': train_min, 'val_max': val_max, 'val_min': val_min},count)
            
            train_results_dataframe = pd.DataFrame(train_results)
            val_results_dataframe = pd.DataFrame(val_results)
            append_or_create(train_results_dataframe, train_results_file)
            append_or_create(val_results_dataframe, val_results_file)

            print(f"Epoch {epoch+1}/{epoch_size}, File {idx+1}/{len(files)} completed.")

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, "max_feature_checkpoint.pt")

    print(count)
    with open("count.txt", "w") as file:
        file.write(str(count))
    torch.save(model.state_dict(), "max_feature_v1.pth")

if __name__ == "__main__":
    main()