# Standard Python libs
import shutil
# Third-party libs
import numpy as np
import pandas as pd
# PyTorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

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
        """
        Model forward pass.

        Parameters:
        - x (torch.Tensor): The input data.

        Returns:
        - torch.Tensor: The concatenated outputs from each fully connected layer.
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 17 * 322)
        
        outputs = [fc(x) for fc in self.fc_layers]
        return torch.cat(outputs, dim=1)
    
def save_ckp(state, is_best):
    """
    Saves the model checkpoint and optionally a copy as the best model. This isfrom Pytorch docs

    Parameters:
    - state (dict): Dictionary containing model's state dict and optimizer's state dict.
    - is_best (bool): If True, saves a copy as the best performing model.
    """
    f_path = 'max_feature_checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = 'max_feature_best_model.pt'
        shutil.copyfile(f_path, best_fpath)


def train_model(model, train_loader, valid_loader, criterion, optimizer, scaler, epoch, device):
    """
    Trains and validates a model, saving checkpoints and returning training/validation results.

    Parameters:
    - model (nn.Module): The neural network model to be trained.
    - train_loader (DataLoader): DataLoader for training data.
    - valid_loader (DataLoader): DataLoader for validation data.
    - criterion (Loss Function): The loss function to use for optimization.
    - optimizer (Optimizer): The optimizer to use for optimization.
    - scaler (GradScaler): Scaler for automatic mixed precision.
    - epoch (int): Current epoch count.
    - device (torch.device): Device to run the model computation on.

    Returns:
    - tuple: A tuple containing dictionaries with training and validation loss and accuracy metrics.
    """
    overall_train_loss = list()
    overall_val_loss = list()
    val_predictions = list()
    val_labels = list()
    model.to(device)

    print("Training started")
    training_loss = 0.0
    validation_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs).float()
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        training_loss += loss.item()

        overall_train_loss.append(training_loss)

        del labels, outputs, loss

    model.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            with autocast():
                outputs = model(inputs).float()
                loss = criterion(outputs, labels)
            validation_loss += loss.item()
            val_predictions.append(outputs.cpu())
            val_labels.append(labels.cpu())
            overall_val_loss.append(validation_loss)

            del labels, outputs, loss


    torch.cuda.empty_cache()

    train_results  = {
        'train_loss' : sum(overall_train_loss) / len(overall_train_loss)
    }
    val_results = {
        'val_loss' : sum(overall_val_loss) / len(overall_val_loss),
    }
    val_labels_batch = {
        'labels' : val_labels,
        'predictions' : val_predictions
    }
    checkpoint = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }
    save_ckp(checkpoint, False)
    return(train_results, val_results, val_labels_batch)