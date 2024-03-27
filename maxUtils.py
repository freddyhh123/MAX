import torch
import os
import pandas as pd
from annoy import AnnoyIndex
from databaseConnector import connect
import csv
import json

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


def update_averages():
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
    db = connect()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM features")
    track_features = cursor.fetchall()

    ann_index = AnnoyIndex(6, 'euclidean')

    for i, features in enumerate(track_features):
        ann_index.add_item(int(features[0]), features[1:7])

    ann_index.build(10)
    ann_index.save('ANN_tracks_index.ann')

update_averages()