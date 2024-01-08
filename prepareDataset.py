import mysql.connector
from databaseConfig import connect
import pandas as pd
import numpy as np
import featureExtraction
import torch

db = connect()
cursor = cursor = db.cursor()


def genres_to_binary(genres, possible_genres):
    return [1 if genre in genres else 0 for genre in possible_genres]

def get_top_genres(track_id, track_genres, genre_to_top_genre):
    sub_genres = track_genres[track_genres['track_id'] == track_id]['genre_id']
    top_genres = set()

    for sub_genre in sub_genres:
        top_genre = genre_to_top_genre.get(sub_genre)
        if top_genre:
            top_genres.add(top_genre)

    return list(top_genres)

def buildDataframe():
    query = "SELECT * FROM tracks LIMIT 10"
    tracks = pd.read_sql(query, db)
    tracks['top_genres'] = np.nan
    tracks['spectrogram'] = np.nan
    tracks['mfcc'] = np.nan
    tracks = tracks.drop(tracks[tracks['track_id'] == '11583'].index)
    tracks = tracks.drop(tracks[tracks['track_id'] == '25173'].index)
    tracks = tracks.drop(tracks[tracks['track_id'] == '25174'].index)

    bad_tracks = pd.read_csv('bad_files.csv')
    bad_tracks['file'] = bad_tracks['file'].astype(str).str.split('.').str[0]
    bad_tracks['file'] = bad_tracks['file'].astype(str).str.lstrip('0')
    bad_track_ids = bad_tracks[bad_tracks['file'].isin(tracks['track_id'])]
    tracks = tracks[~tracks['track_id'].isin(bad_track_ids['file'])]

    query = "SELECT genre_id FROM genres WHERE genre_parent = 0"
    cursor.execute(query)
    all_top_genres = cursor.fetchall()
    all_top_genres = [item[0] for item in all_top_genres]

    genre_relationship_query = "SELECT genre_id, top_genre FROM genres"
    genre_relationships_df = pd.read_sql(genre_relationship_query, db)

    track_genre_query = "SELECT track_id, genre_id FROM track_genres"
    track_genres_df = pd.read_sql(track_genre_query, db)

    genre_to_top_genre = {row[1]: row[2] for row in genre_relationships_df.itertuples()}

    tracks['top_genres'] = tracks['track_id'].apply(get_top_genres, args=(track_genres_df,genre_to_top_genre))

    tracks['genre_vector'] = tracks['top_genres'].apply(lambda x: genres_to_binary(x, all_top_genres))

    tracks['spectrogram'] = tracks['track_id'].apply(featureExtraction.gen_spectrogram)

    tracks['mfcc'] = tracks['track_id'].apply(featureExtraction.gen_mffc)
    
    print("Dataset finished")

    return tracks
    

