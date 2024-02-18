import mysql.connector
from databaseConfig import connect
import csv
import uuid
import time
import pandas as pd
import json
import math
# This is from the FMA Github repository, it is used to
# properly remove some features from the dataset
import utils
import torchaudio

db = connect()

cursor = db.cursor()

def test_audio(file):
    try:
        file_path = utils.get_audio_path("data",int(file))
        wav, sample_rate = torchaudio.load(file_path, normalize = True)
        return True
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return False

def addTracks():
    # Start from scratch, if we are running this, everything is reset!
    with open("MAXschema.sql", 'r') as schema:
        script = schema.read()
    commands = script.split(';')
    for command in commands:
        try:
            if command.strip():
                cursor.execute(command)
        except Exception as e:
            print(f"Error executing command: {command}")
            print(e)
            break
    db.commit()

    batchId = str(uuid.uuid1())
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Populate tables from csvs
    populateArtistTable()
    populateAlbumTable()
    populateGenreTable()

    # We need to do our features first as there is a foreign key requirement later
    # This is also from the FMA dataset utils file!
    echonest = utils.load('fma_data/echonest.csv')
    features = echonest['echonest','audio_features']
    for row in features.iterrows():
        sql = "INSERT INTO features (featureset_id, danceability, energy, speechiness, acousticness, instrumentalness, liveleness, valence, tempo) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        values = (row[0], row[1]['danceability'], row[1]['energy'], row[1]['speechiness'], row[1]['acousticness'], row[1]['instrumentalness'], row[1]['liveness'], row[1]['valence'], row[1]['tempo'])
        cursor.execute(sql, values)
    db.commit()

    # Remove some columns for efficiency!
    columns_to_remove = [
    "album_url", "artist_name", "artist_url", "artist_website",
    "license_image_file", "license_parent_id", "track_comments", 
    "track_composer", "track_copyright_c", "track_copyright_p", 
    "track_date_created", "track_disc_number", "track_explicit_notes", 
    "track_favorites", "track_file", "track_information", 
    "track_language_code", "track_lyricist", "track_publisher"
    ]
    tracks = pd.read_csv('fma_data/raw_tracks.csv')
    tracks.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    # Drop tracks with bad audio files (found using check_files.py)
    bad_tracks = pd.read_csv('bad_files.csv')
    bad_tracks['file'] = bad_tracks['file'].astype(str).str.split('.').str[0]
    bad_tracks['file'] = bad_tracks['file'].astype(str).str.lstrip('0')
    bad_track_ids = bad_tracks[bad_tracks['file'].isin(tracks['track_id'])]
    tracks = tracks[~tracks['track_id'].isin(bad_track_ids['file'])]

    for track in tracks.iterrows():
        # Make sure there is a genre as that is vital! we do a similar check later for album and artist
        # but as they are not as important to analysis we can keep the tracks and add unknown artist!
        if type(track[1]['track_genres']) != float and type(track[1]["track_title"]) != float:
            # Make sure we dont have the track (just in case)
            if checkExisting("tracks", "track_id", track[0]) == False:
                # Insert out track
                sql = "INSERT INTO tracks (track_id,track_name,file_path,time_added,trained,batch) VALUES (%s,%s,%s,%s,%s,%s)"
                # Note, using get_audio_path from the FMA dataset utils code!
                if test_audio(int(track[0])) == False:
                    continue
                values = (track[0], track[1]["track_title"], utils.get_audio_path("data",int(track[0])), timestamp, False, batchId)
                cursor.execute(sql, values)
                print("Added " + str(track[1]["track_title"]) + " to track table")

                # Format our genres from json and add!
                genres = json.loads(track[1]['track_genres'].replace("'", '"'))
                for genre in genres:
                    sql = "INSERT INTO track_genres (track_id, genre_id) VALUES (%s, %s)"
                    values = (track[0], int(genre["genre_id"]))
                    try:
                        cursor.execute(sql, values)
                    except Exception as e:
                        print("Error - missing genre")
                        continue
                
                # For Artists and albums make sure all data is there and insert!
                if type(track[1]["artist_id"]) != float and checkExisting("artists","artist_id", int(track[1]["artist_id"])) == True:
                    sql = "INSERT INTO artists_tracks (artist_id, track_id) VALUES (%s,%s)"
                    values = (track[1]["artist_id"], track[0])
                    cursor.execute(sql, values)
                else:
                    sql = "INSERT INTO artists_tracks (artist_id, track_id) VALUES (%s, %s)"
                    values = (101010, track[0])
                    cursor.execute(sql, values)
                
                if type(track[1]['album_id']) != float and checkExisting("albums", "album_id", int(track[1]['album_id'])) == True:
                    sql = "INSERT INTO album_tracks (track_id, album_id) VALUES (%s, %s)"
                    values = (track[0], int(track[1]['album_id']))
                    cursor.execute(sql, values)
                else:
                    sql = "INSERT INTO album_tracks (track_id, album_id) VALUES (%s, %s)"
                    values = (track[0], 101010)
                    cursor.execute(sql, values)
    db.commit()

# Helper function to check if something exists in a table
def checkExisting(table, key, value):
    sql = "SELECT * FROM " + table + " WHERE "+ key + " = %s"
    values = (value, )
    cursor.execute(sql, values)
    existing = cursor.fetchall()
    if existing:
        return True
    else:
        return False

# Helper function to add Artist
def addArtist(id, name, url):
    existing = checkExisting("artists", "artist_id", id)
    if not existing:
        sql = "INSERT INTO artists VALUES (%s,%s,%s)"
        values = (id, name, url)
        cursor.execute(sql, values)
        print("Added " + str(name + " to artist table"))

# Functions to populate various tables, note, these add an "unknown" option for any missing info!
def populateGenreTable():
    with open('fma_data/genres.csv', 'r',encoding='utf-8') as genres:
        reader = csv.reader(genres)
        next(reader)
        for i, genre in enumerate(reader, start=1):
            cursor.execute("SELECT * FROM genres WHERE genre_id = %s", (genre[0],))
            exists = cursor.fetchone()
            if exists:
                print("This genre ID exists.")
            else:
                sql = "INSERT INTO genres (genre_id, genre_parent, top_genre, genre_name) VALUES (%s,%s,%s,%s)"
                values = (genre[0],genre[2],genre[4],genre[3])
                cursor.execute(sql, values)
    print("Genres added")
    db.commit()

def populateAlbumTable():
    sql = "INSERT INTO albums (album_id, album_link, album_name) VALUES (%s,%s,%s)"
    values = (101010, "", "Unknown")
    cursor.execute(sql, values)
    with open('fma_data/raw_albums.csv', 'r',encoding='utf-8') as albums:
        reader = csv.reader(albums)
        next(reader)
        for i, album in enumerate(reader, start=1):
            cursor.execute("SELECT * FROM albums WHERE album_id = %s", (album[0],))
            exists = cursor.fetchone()
            if exists:
                print("This album ID exists.")
            else:
                sql = "INSERT INTO albums (album_id, album_link, album_name) VALUES (%s,%s,%s)"
                values = (album[0], album[15], album[6])
                cursor.execute(sql, values)
                cursor.execute("SELECT artist_id FROM artists WHERE artist_name = %s", (album[16],))
                exists = cursor.fetchone()
                if exists:
                    sql = "INSERT INTO artists_albums (artist_id, album_id) VALUES (%s,%s)"
                    values = (exists[0], album[0])
                    cursor.execute(sql, values)
    print("Albums Added")
    db.commit()

def populateArtistTable():
    sql = "INSERT INTO artists (artist_id, artist_name, artist_url) VALUES (%s,%s,%s)"
    values = (101010, "Unknown","https://freemusicarchive.org/")
    cursor.execute(sql, values)
    with open('fma_data/raw_artists.csv', 'r',encoding='utf-8') as artists:
        reader = csv.reader(artists)
        next(reader)
        for i, artist in enumerate(reader, start=1):
            cursor.execute("SELECT * FROM artists WHERE artist_id = %s", (artist[0],))
            exists = cursor.fetchone()
            if exists:
                print("This artist ID exists.")
            else:
                sql = "INSERT INTO artists (artist_id, artist_name, artist_url) VALUES (%s,%s,%s)"
                values = (artist[0], artist[11], artist[21])
                cursor.execute(sql, values)
    print("Artists Added")
    db.commit()