import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy_random import get_random

import databaseConnector

import spectrogram


def getTracks():
    cid = "516f5defb70e4e0994acea4cc99d1d4c"
    secret = "0038b1fca65d467dbd0d2e1437041545"

    client_credentials_manager = SpotifyClientCredentials(
    client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout=10, retries=10)

    trackWithAudio = []

    for i in range(5):
        try:
            track = get_random(spotify=sp, type="track")
        except:
            print('Spotify done goofed, re-arm!')
            track = get_random(spotify=sp, type="track")

        if track['preview_url']:
            compiledTrack = {}
            compiledAlbum = {}
            
            compiledTrack['album'] = {}

            compiledArtists = [{}]
            compiledTrack['artists'] = []
            compiledAlbum['artists'] = []
            for key, artist in enumerate(track['artists']):
                compiledArtist = {}

                compiledTrack["genres"] = sp.artist(artist["id"])["genres"]

                compiledArtist["artist_url"] = artist['external_urls']['spotify']
                compiledArtist["name"] = artist['name']
                compiledArtist["id"] = artist['id']

                compiledArtists.append(compiledArtist)
                compiledTrack["artists"].append(compiledArtist["id"])

            compiledTrack['name'] = track['name']
            compiledTrack['id'] = track['id']
            compiledTrack['album'] = track['album']['id']
            compiledTrack['preview_url'] = track['preview_url']
            compiledTrack['spotify_url'] = track['external_urls']['spotify']
            compiledTrack['features'] = sp.audio_features(track["id"])[0]
            
            compiledAlbum['name'] = track['album']['name']
            compiledAlbum['id'] = track['album']['id']
            compiledAlbum['url'] = track['album']['external_urls']['spotify']
            for key, artist in enumerate(track['album']['artists']):
                if artist not in compiledArtists:
                    compiledArtist = {}
                    compiledArtist["artist_url"] = artist['external_urls']['spotify']
                    compiledArtist["name"] = artist['name']
                    compiledArtist["id"] = artist['id']

                    compiledArtists.append(compiledArtist)
                compiledAlbum["artists"].append(compiledArtist["id"])

                compiledTrack = spectrogram.gen_spectrogram(compiledTrack)
            
            trackWithAudio.append(compiledTrack)
    return trackWithAudio


tracks = getTracks()
databaseConnector.addTracks(tracks)

            





