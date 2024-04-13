import torch
import os
import pandas as pd
from annoy import AnnoyIndex
from databaseConnector import connect
import csv
import json
import torchaudio
from subGenreModel import SubGenreClassifier

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    Load a model checkpoint. This is from pytorch docs.

    Parameters:
    checkpoint_fpath (str): Path to the checkpoint file.
    model (torch.nn.Module): The model that needs to be loaded.
    optimizer (torch.optim.Optimizer): The optimizer that needs to be loaded.

    Returns:
    tuple: Returns the updated model and optimizer loaded with checkpoint data.
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def append_or_create(results_dataframe, file_path):
    """
    Append a dataframe to an existing CSV file or create a new CSV file if it doesn't exist.

    Parameters:
    results_dataframe (pd.DataFrame): Dataframe containing the results to append.
    file_path (str): Path to the CSV file.

    Returns:
    None
    """
    if os.path.exists(file_path):
        existing_dataframe = pd.read_csv(file_path)
        updated_dataframe = pd.concat([existing_dataframe, results_dataframe], ignore_index=True)
    else:
        updated_dataframe = results_dataframe

    updated_dataframe.to_csv(file_path, index=False)


def update_averages():
    """
    Update genre averages

    Returns:
    None
    """
    db = connect()
    cursor = db.cursor(dictionary=True)
    query = """
    SELECT 
        g.genre_name,
        g.genre_id,
        AVG(f.danceability) AS avg_danceability,
        AVG(f.energy) AS avg_energy,
        AVG(f.speechiness) AS avg_speechiness,
        AVG(f.acousticness) AS avg_acousticness,
        AVG(f.instrumentalness) AS avg_instrumentalness,
        AVG(f.liveleness) AS avg_liveleness,
        AVG(f.valence) AS avg_valence
    FROM 
        genres g
    JOIN 
        track_genres tg ON g.genre_id = tg.genre_id
    JOIN 
        tracks t ON tg.track_id = t.track_id
    JOIN 
        features f ON t.track_id = f.featureset_id
    WHERE 
        g.genre_parent = 0
    GROUP BY 
        g.genre_name, g.genre_id;
    """
    cursor.execute(query)
    averages = cursor.fetchall()
    genre_features = [[row['genre_id']] + [row['genre_name']] + [float(row['avg_danceability']), float(row['avg_energy']), float(row['avg_speechiness']), float(row['avg_acousticness']), float(row['avg_instrumentalness']), float(row['avg_liveleness']), float(row['avg_valence'])] for row in averages]
    genre_features = json.dumps(genre_features)

    with open('static/data/genre_averages.json', 'w') as json_file:
        json_file.write(genre_features)


def update_ann():
    """
    Build and save an ANN for nearest track.

    Returns:
    None
    """
    db = connect()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM features")
    track_features = cursor.fetchall()

    ann_index = AnnoyIndex(6, 'euclidean')

    for i, features in enumerate(track_features):
        ann_index.add_item(int(features[0]), features[1:7])

    ann_index.build(10)
    ann_index.save('ANN_tracks_index.ann')

def check_bad_files():
    """
    Check for corrupted MP3 files in a specified directory and log.

    Returns:
    None
    """
    directory = 'data'
    bad_files = []
    fields = ['folder', 'file', 'error'] 

    for folder in os.listdir(directory):
        if folder == "checksums":
            continue
        for filename in os.listdir(os.path.join(directory, folder)):
            if filename.endswith('.mp3'):
                full_path = os.path.join(directory, folder, filename)
                try:
                    torchaudio.load(full_path, normalize=True)
                except Exception as e:  # Catch the exception to understand the issue
                    bad_files.append((folder, filename, str(e)))

    with open('bad_files.csv', 'w',newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(bad_files)

def extract_parameters_from_filename(filename):
    """
    Extracts parameters from a given filename based on expected format.

    Parameters:
    - filename (str): Filename containing epoch size, batch size, and learning rate.

    Returns:
    - tuple: Contains extracted epoch size, batch size, and learning rate.
    """
    parts = filename.split('_')
    params = parts[0].split('-')
    epoch_size, batch_size, learning_rate = params[:3]
    return epoch_size, batch_size, learning_rate

def initialize_sub_genre_models():
    """
    Initialized sub genre models with correct output sizes.

    Returns:
    - List: Contains models.
    """
    # info for generating sub-genre models, left side is genre_id
    # right side is the number of sub_genres or outputs
    sub_genre_models = {}
    sub_genre_info = [30,2,7,8,1,6,3,29,4,4,19,6,12,8,19,5]
    for idx, sub_genres in enumerate(sub_genre_info):
                    sub_model = SubGenreClassifier(num_classes = sub_genres)
                    sub_genre_models[idx] = sub_model
    return sub_genre_models

def get_top_to_sub_genre_map():
    """
    Gets map of sub-genres to top genres.

    Returns:
    - List: Contains map.
    """
    db = connect()
    genre_relationship_query = "SELECT genre_id, top_genre FROM genres"
    genre_relationships_df = pd.read_sql(genre_relationship_query, db)
    return {row['genre_id']: row['top_genre'] for index, row in genre_relationships_df.iterrows()}
