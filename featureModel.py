import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import shutil

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
        
        self.fc_layers = nn.ModuleList([nn.Linear(64 * 17 * 322, 1) for _ in range(num_features)])

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 17 * 322)
        
        outputs = [fc(x) for fc in self.fc_layers]
        return torch.cat(outputs, dim=1)
    
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
    return model, optimizer, checkpoint['epoch']
    

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
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            optimizer.zero_grad()
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
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
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
    checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }
    save_ckp(checkpoint, False)
    return(train_results, val_results)