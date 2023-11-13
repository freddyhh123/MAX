import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy_random import get_random
import uuid
import time

import databaseConnector

import spectrogram


def getTracks():
    cid = "516f5defb70e4e0994acea4cc99d1d4c"
    secret = "0038b1fca65d467dbd0d2e1437041545"

    client_credentials_manager = SpotifyClientCredentials(
    client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout=10, retries=10)

    trackWithAudio = []

    batchId = str(uuid.uuid1())
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    while len(trackWithAudio) < 5:
        try:
            track = get_random(spotify=sp, type="track")
        except:
            print('Spotify done goofed, re-arm!')
            track = get_random(spotify=sp, type="track")

        if track['preview_url']:
            compiledTrack = {}
            compiledAlbum = {}
            
            compiledTrack['album'] = {}

            compiledArtists = []
            compiledTrack['artists'] = []
            compiledAlbum['artists'] = []
            for key, artist in enumerate(track['artists']):
                compiledArtist = {}

                compiledTrack["genres"] = sp.artist(artist["id"])["genres"]

                compiledArtist["artist_url"] = artist['external_urls']['spotify']
                compiledArtist["artist_name"] = artist['name']
                compiledArtist["artist_id"] = artist['id']

                compiledArtists.append(compiledArtist)
                compiledTrack["artists"].append(compiledArtist["artist_id"])

            compiledTrack['timestamp'] = timestamp

            compiledTrack['name'] = track['name']
            compiledTrack['id'] = track['id']
            compiledTrack['album'] = track['album']['id']
            compiledTrack['preview_url'] = track['preview_url']
            compiledTrack['spotify_url'] = track['external_urls']['spotify']

            compiledTrack['features'] = sp.audio_features(track["id"])[0]
            if not compiledTrack['features']:
                continue
            compiledTrack['features'].pop('type',None)
            compiledTrack['features'].pop('id',None)
            compiledTrack['features'].pop('track_href',None)
            compiledTrack['features'].pop('analysis_url',None)
            compiledTrack['features'].pop('time_signature',None)
            compiledTrack['features'].pop('uri',None)

            compiledAlbum['name'] = track['album']['name']
            compiledAlbum['id'] = track['album']['id']
            compiledAlbum['url'] = track['album']['external_urls']['spotify']
            compiledAlbum['images'] = track['album']['images']
            for key, artist in enumerate(track['album']['artists']):
                if artist not in compiledArtists:
                    compiledArtist = {}
                    compiledArtist["artist_url"] = artist['external_urls']['spotify']
                    compiledArtist["artist_name"] = artist['name']
                    compiledArtist["artist_id"] = artist['id']

                    compiledArtists.append(compiledArtist)
                compiledAlbum["artists"].append(compiledArtist)

                compiledTrack['batchId'] = batchId
                compiledTrack = spectrogram.gen_spectrogram(compiledTrack)
            
            compiledTrack['album'] = compiledAlbum
            compiledTrack['artists'] = compiledArtists
            
            trackWithAudio.append(compiledTrack)
    return trackWithAudio


tracks = getTracks()
databaseConnector.addTracks(tracks)
