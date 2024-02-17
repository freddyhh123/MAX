from databaseConfig import connect
import pandas as pd
import numpy as np
from featureExtraction import gen_spectrogram
from featureExtraction import gen_mfcc
import pickle
import random

# Set db connection for whole file, not best practice but we are only running this one at a time
db = connect()
cursor = cursor = db.cursor()

# Converting out genres into one hot encoding
def genres_to_binary(genres, possible_genres):
    return [1 if genre in genres else 0 for genre in possible_genres]

# This gets a list of to level genres for a track
def get_top_genres(track_id, track_genres_df, genre_to_top_genre):
    cursor.execute("SELECT track_id, genre_id FROM track_genres WHERE track_id = %s;",(track_id,))
    sub_genres = cursor.fetchall()
    top_genres = set()

    for sub_genre in sub_genres:
        top_genre = genre_to_top_genre.get(sub_genre[1])
        if top_genre:
            top_genres.add(top_genre)

    return list(top_genres)

# Gets all available top level genres
def get_all_top_genres():
    query = "SELECT genre_id FROM genres WHERE genre_parent = 0"
    cursor.execute(query)
    all_top_genres = cursor.fetchall()
    return(all_top_genres)

# Getting all the names for top level genres
def all_top_genre_names():
    query = "SELECT genre_id FROM genres WHERE genre_parent = 0"
    cursor.execute(query)
    top_genre_ids = cursor.fetchall()
    top_genre_ids = [item[0] for item in top_genre_ids]
    genres = list()
    for genre in top_genre_ids:
        cursor.execute('SELECT genre_name, genre_id FROM genres WHERE genre_id = %s',(genre,))
        genres.append(cursor.fetchone())
    return(genres)

# Find the featureset for a track
def get_track_features(track_id):
    query = "SELECT * FROM features WHERE featureset_id = %s"
    values = (track_id,)
    cursor.execute(query,values)
    features = cursor.fetchone()
    return(features[1:9])

# This processes and saves a chunk of tracks to a file
def get_tracks(dataset_type, tracks_chunk, number):
    if dataset_type == "features":
        columns = ['track_id'] + ['track_name'] + ['file_path'] + ['time_added'] + ['trained'] + ['batch']
    else:
        columns = ['index'] + ['track_id'] + ['track_name'] + ['file_path'] + ['time_added'] + ['trained'] + ['batch'] + ["num"]
    tracks = pd.DataFrame(tracks_chunk, columns = columns)
    tracks = tracks[['track_id', 'track_name', 'file_path']]
    
    # Make some empty columns for later use
    tracks['spectrogram'] = np.nan
    tracks['mfcc'] = np.nan

    # If its features this is nice and easy!
    if dataset_type == "features":
        tracks['features'] = np.nan if dataset_type == "features" else None
        tracks['features'] = tracks['track_id'].apply(get_track_features)
    else:
        tracks['top_genres'] = np.nan if dataset_type != "features" else None
        tracks['genre_vector'] = np.nan if dataset_type != "features" else None
        # Getting all top genres - Refer to schema if this is confusing
        all_top_genres = get_all_top_genres()
        all_top_genres = [item[0] for item in all_top_genres]
        genre_relationship_query = "SELECT genre_id, top_genre FROM genres"
        genre_relationships_df = pd.read_sql(genre_relationship_query, db)
        # Getting track genre info
        track_genre_query = "SELECT track_id, genre_id FROM track_genres"
        track_genres_df = pd.read_sql(track_genre_query, db)
        # Converting low level genres to their relevant top level genres
        genre_to_top_genre = {row['genre_id']: row['top_genre'] for index, row in genre_relationships_df.iterrows()}
        tracks['top_genres'] = tracks['track_id'].apply(get_top_genres, args=(track_genres_df, genre_to_top_genre))
        # Converting this to one hot encoding
        tracks['genre_vector'] = tracks['top_genres'].apply(lambda x: genres_to_binary(x, all_top_genres))

    # Generating audio features, this is VERY resource intensive
    tracks['spectrogram'] = tracks['track_id'].apply(gen_spectrogram)
    tracks['mfcc'] = tracks['track_id'].apply(gen_mfcc)

    # Saving for processing later on
    filename = f"tracks-{number}.pkl"
    with open(filename, "wb") as file:
        pickle.dump(tracks, file)

def buildDataframe(dataset_type, split):
    # Checking if we are splitting the data
    if split:
        # Features are easy!
        if dataset_type == "features":
            cursor.execute("""
                SELECT tracks.*
                FROM tracks
                INNER JOIN features ON tracks.track_id = features.featureset_id;
            """)
            tracks = cursor.fetchall()
            random.shuffle(tracks)
            tracks_per_file = 1000
            for i in range(0, len(tracks), tracks_per_file):
                chunk = tracks[i:i+tracks_per_file]
                file_number = i // tracks_per_file + 1
                get_tracks(dataset_type, chunk, file_number)
        else:
            # This query gets the top level genre with the least tracks
            # we can then use this for balancing
            cursor.execute("""
                SELECT MIN(genre_count) AS min_tracks
                FROM (
                    SELECT COUNT(t.track_id) AS genre_count
                    FROM tracks t
                    JOIN track_genres tg ON t.track_id = tg.track_id
                    JOIN genres g ON tg.genre_id = g.genre_id
                    GROUP BY g.top_genre
                ) AS genre_counts;
            """)
            min_tracks = cursor.fetchone()[0]
            # This now gets a sample of every genre but with the exact same amount for every genre
            cursor.execute(f"""
                WITH RankedTracks AS (
                    SELECT
                        g.top_genre,
                        t.*,
                        ROW_NUMBER() OVER (PARTITION BY g.top_genre ORDER BY t.track_id) AS genre_rank
                    FROM tracks t
                    JOIN track_genres tg ON t.track_id = tg.track_id
                    JOIN genres g ON tg.genre_id = g.genre_id
                )
                SELECT *
                FROM RankedTracks
                WHERE genre_rank <= {min_tracks}
            """)
            tracks = cursor.fetchall()
        # The shuffle is important!
        random.shuffle(tracks)

        tracks_per_file = 1000
        for i in range(0, len(tracks), tracks_per_file):
            chunk = tracks[i:i+tracks_per_file]
            file_number = i // tracks_per_file + 1
            get_tracks(dataset_type, chunk, file_number)

    else:
        # If no split is needed, fetch all tracks and process in one go
        cursor.execute("SELECT * FROM tracks")
        tracks = cursor.fetchall()
        get_tracks(dataset_type, tracks, 0)