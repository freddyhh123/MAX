import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
import pandas as pd
import numpy as np
import shutil

class SubGenreClassifier(nn.Module):
    def __init__(self, input_channels=2, num_classes=16):
        super(SubGenreClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        # This is the size of the flattened features and the size of this layer
        self.fc1 = nn.Linear(92736, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x =x.view(x.shape[0], 92736)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def save_ckp(state, is_best):
    f_path = 'max_genre_checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = 'max_genre_best_model.pt'
        shutil.copyfile(f_path, best_fpath)
    

def train_sub_models(sub_genre_models, train_loader, valid_loader, criterion, epoch, device):
    sigmoid = torch.nn.Sigmoid()

    print("Training started")
    correct_train = 0.0
    correct_val = 0.0
    total_train = 0.0
    total_val = 0.0

    for inputs, genre_info in train_loader:
        for i in range(inputs.size(0)):
            input_item = inputs[i].unsqueeze(0).to(device)
            
            sub_genres = genre_info[i]['sub_genre']
            for sub_genre in sub_genres:
                model = sub_genre_models[sub_genre]['model'].to(device)
                optimizer = sub_genre_models[sub_genre]['optomizer']
                
                model.train()
                optimizer.zero_grad()
                
                output = model(input_item)
                label = torch.tensor(sub_genres[sub_genre]).unsqueeze(0).float().to(device)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                probabilities = sigmoid(output.data)
                predictions = (probabilities > 0.3).int()
                total_train += label.size(0)
                correct_train = (predictions == label).float()
                train_sample_accuracy = correct_train.mean(dim=1)
                sub_genre_models[sub_genre]['train_accuracy'].append(train_sample_accuracy.mean().item())
                
                sub_genre_models[sub_genre]['train_loss'] += loss.item()

                sub_genre_models[sub_genre]['count'] += 1

    model.eval()
    with torch.no_grad():
        for inputs, genre_info in valid_loader:
            for i in range(inputs.size(0)):
                input_item = inputs[i].unsqueeze(0).to(device)
                
                sub_genres = genre_info[i]['sub_genre']
                for sub_genre in sub_genres:
                    model = sub_genre_models[sub_genre]['model'].to(device)
                    optimizer = sub_genre_models[sub_genre]['optomizer']
                    
                    model.train()
                    optimizer.zero_grad()
                    
                    output = model(input_item)
                    label = torch.tensor(sub_genres[sub_genre]).unsqueeze(0).float().to(device)
                    loss = criterion(output, label)

                    probabilities = sigmoid(output.data)
                    predictions = (probabilities > 0.3).int()
                    total_val += label.size(0)
                    correct_val = (predictions == label).float()
                    val_sample_accuracy = correct_val.mean(dim=1)
                    sub_genre_models[sub_genre]['val_accuracy'].append(val_sample_accuracy.mean().item())
                    
                    sub_genre_models[sub_genre]['val_loss'] += loss.item()

                    sub_genre_models[sub_genre]['count'] += 1

    checkpoint = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }
    save_ckp(checkpoint, False)
    return(sub_genre_models)