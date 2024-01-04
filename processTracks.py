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

def getTracks():

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
            if i > 10:
                break

            if track[19]:
                compiledTrack = {}

                compiledTrack['timestamp'] = timestamp

                compiledTrack['name'] = track[2]
                compiledTrack['id'] = track[0]
                compiledTrack['album'] = track[1]
                compiledTrack['artists'] = track[3]
                #compiledTrack['preview_url'] = utils.get_audio_path("data",track[0])
                compiledTrack['file_path'] = "test"
                track_features = features[features['track_id'] == int(track[0])]
                compiledTrack['features'] = track_features.to_dict()
                if not compiledTrack['features']:
                    continue
                
                trackWithAudio.append(compiledTrack)
    return trackWithAudio

tracks = getTracks()
databaseConnector.addTracks(tracks)
