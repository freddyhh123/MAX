import torch
import os
import pandas as pd

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