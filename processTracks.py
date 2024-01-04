import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy_random import get_random
import uuid
import time
from databaseConfig import connect
import databaseConnector
import featureExtraction
import csv
import utils
import json


def unpack_features(features):
    values = {}
    for outer_key, inner_dict in features.items():
        for key in inner_dict:
            values[outer_key] = inner_dict[key]
    return values


def getTracks(batch_size):
    with open('refined_tracks.csv', 'r',encoding='utf-8') as tracks:
        features = pd.read_csv('refined_analysis.csv')
        reader = csv.reader(tracks)
        next(reader)

        trackWithAudio = []

        db = connect()
        cursor = db.cursor()

        batchId = str(uuid.uuid1())
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print("Batch id: " + batchId)
        for i, track in enumerate(reader, start=1):
            if batch_size is not None:
                if i > batch_size:
                    break

            if track[19]:
                compiledTrack = {}

                compiledTrack['timestamp'] = timestamp

                compiledTrack['name'] = track[18]
                compiledTrack['id'] = track[0]
                compiledTrack['album'] = track[1]
                compiledTrack['artist'] = track[3]
                compiledTrack['genres'] = list()
                genre_info = json.loads(track[12].replace("'", '"'))
                for genre in genre_info:
                    compiledTrack['genres'].append(genre['genre_id'])
                compiledTrack['file_path'] = utils.get_audio_path("data",int(track[0]))
                track_features = features[features['track_id'] == int(track[0])]
                compiledTrack['features'] = unpack_features(track_features.to_dict())
                if not compiledTrack['features']:
                    continue
                compiledTrack['batchId'] = batchId
                trackWithAudio.append(compiledTrack)
    return trackWithAudio
