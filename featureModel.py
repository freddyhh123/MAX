import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class audioFeatureModel(nn.Module):
    def __init__(self, input_channels=2, num_features=8):
        super(audioFeatureModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 17 * 322, 256)
        self.fc2 = nn.Linear(256, num_features)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 17 * 322)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x)
        return x
    

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, normalization_values):
    overall_train_loss = list()
    overall_val_loss = list()
    model.to(device)

    for epoch in range(num_epochs):
        print("Training started")
        training_loss = 0.0
        validation_loss = 0.0
        model.train()
        for inputs, labels in train_loader:
            labels = torch.stack(labels)
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            optimizer.zero_grad()
            labels = labels.view(1,8)
            if inputs.shape[1] == 1:
                inputs = inputs.repeat(1, 2, 1, 1)
            outputs = model(inputs).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        overall_train_loss.append(training_loss / len(train_loader))

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {training_loss / len(train_loader)}")

        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                labels = torch.stack(labels)
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                labels = labels.view(1,8)
                if inputs.shape[1] == 1:
                    inputs = inputs.repeat(1, 2, 1, 1)
                outputs = model(inputs).float()
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
            overall_val_loss.append(validation_loss / len(valid_loader))
            print(f"Validation Loss: {validation_loss / len(valid_loader)}")

    train_results = pd.DataFrame({
        'train_loss': overall_train_loss
    })
    val_results = pd.DataFrame({
        'val_loss': overall_val_loss
    })
    torch.save(model.state_dict(), 'max_v1.pth')
    return(train_results, val_results)