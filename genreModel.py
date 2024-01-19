import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
import pandas as pd
import numpy as np
import shutil

class topGenreClassifier(nn.Module):
    def __init__(self, input_channels=2, num_classes=16):
        super(topGenreClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(722400, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 32 * 35 * 645)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def save_ckp(state, is_best):
    f_path = 'max_feature_checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = 'max_feature_best_model.pt'
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
    

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    sigmoid = torch.nn.Sigmoid()
    overall_train_loss = list()
    overall_val_loss = list()
    f1_macro_scores = list()
    f1_micro_scores = list()
    hammingloss = list()
    subset_accuracy = list()
    epoch_train_accuracy = list()
    epoch_validation_accuracy = list()
    model.to(device)

    for epoch in range(num_epochs):
        print("Training started")
        training_loss = 0.0
        validation_loss = 0.0
        correct_train = 0.0
        correct_val = 0.0
        total_train = 0.0
        total_val = 0.0
        model.train()
        for inputs, labels in train_loader:
            inputs,labels = inputs.to(device), labels.to(device)
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)

            probabilities = sigmoid(outputs.data)
            predictions = (probabilities > 0.5).int()   
            total_train += labels.size(0)
            correct_train = (predictions == labels).float()
            train_sample_accuracy = correct_train.mean(dim=1)
            train_accuracy = train_sample_accuracy.mean().item()

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            training_loss += loss.item()

        overall_train_loss.append(training_loss / len(train_loader))
        epoch_train_accuracy.append(train_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {training_loss / len(train_loader)}")

        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs,labels = inputs.to(device), labels.to(device)
                labels = labels.float()
                outputs = model(inputs)

                probabilities = sigmoid(outputs.data)
                predictions = (probabilities > 0.5).int()   

                total_val += labels.size(0)
                correct_val = (predictions == labels).float()
                val_sample_accuracy = correct_val.mean(dim=1)
                val_accuracy = val_sample_accuracy.mean().item()

                loss = criterion(outputs, labels)
                validation_loss += loss.item()
            
            if device.type == "cuda":
                f1_macro_scores.append(f1_score(labels.cpu(), predictions.cpu(), average='macro'))
                f1_micro_scores.append(f1_score(labels.cpu(), predictions.cpu(), average='micro'))
                hammingloss.append(hamming_loss(labels.cpu(), predictions.cpu()))
                subset_accuracy.append(accuracy_score(labels.cpu(), predictions.cpu()))
            else:
                f1_macro_scores.append(f1_score(labels, predictions, average='macro'))
                f1_micro_scores.append(f1_score(labels, predictions, average='micro'))
                hammingloss.append(hamming_loss(labels, predictions))
                subset_accuracy.append(accuracy_score(labels, predictions))

            overall_val_loss.append(validation_loss / len(valid_loader))
            epoch_validation_accuracy.append(val_accuracy)

        print(f"Validation Loss: {validation_loss / len(valid_loader)}")

    train_results  = pd.DataFrame ({
        'train_accuracy': epoch_train_accuracy,
        'train_loss' : overall_train_loss
    })
    val_results  = pd.DataFrame ({
        'val_accuracy': epoch_validation_accuracy,
        'val_loss' : overall_val_loss,
        'f1_macro' : f1_macro_scores,
        'f1_micro' : f1_micro_scores,
        'hamming_loss' : hammingloss,
        'subset_accuracy' : subset_accuracy
    })
    checkpoint = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }
    save_ckp(checkpoint, False)
    return(train_results, val_results)