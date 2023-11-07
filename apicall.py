import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials

import torch
import torchaudio

import urllib.request
from pydub import AudioSegment

def getTracks():
    cid = "516f5defb70e4e0994acea4cc99d1d4c"
    secret = "0038b1fca65d467dbd0d2e1437041545"

    client_credentials_manager = SpotifyClientCredentials(
    client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    playlist_uri = 'spotify:playlist:44laAM2s0CKDb9AdBpKtfZ'

    results = sp.playlist_items(playlist_uri)
    trackWithAudio = []

    for entry in results['items'][:250]:
        track = entry["track"]
        if track['preview_url']:
            compiledTrack = {}
            compiledArtist = {}
            compiledTrack['album'] = {}
            compiledTrack['artists'] = []
            for key, artist in enumerate(track['artists']):
                compiledTrack["genres"] = sp.artist(artist["id"])["genres"]

                compiledArtist["artist_url"] = artist['external_urls']['spotify']
                compiledArtist["name"] = artist['name']
                compiledArtist["id"] = artist['id']
                compiledTrack["artists"].append(compiledArtist)

            compiledTrack['name'] = track['name']
            compiledTrack['id'] = track['id']
            compiledTrack['popularity'] = track['popularity']
            compiledTrack['preview_url'] = track['preview_url']
            compiledTrack['spotify_url'] = track['external_urls']['spotify']
            compiledTrack['spotify_id'] = track['uri']
            compiledTrack['features'] = sp.audio_features(track["id"])[0]

            compiledTrack['explicit'] = track['explicit']
            
            compiledTrack['album']['name'] = track['album']['name']
            compiledTrack['album']['id'] = track['album']['id']
            compiledTrack['album']['type'] = track['album']['album_type']
            compiledTrack['album']['url'] = track['album']['external_urls']['spotify']
            compiledTrack['album']['image'] = track['album']['images']
            compiledTrack['album']['release_date'] = track['album']['release_date']

            urllib.request.urlretrieve(compiledTrack['preview_url'], "audio/" + str(compiledTrack['id']) + ".mp3")
            wav = AudioSegment.from_mp3("audio/" + str(compiledTrack['id']) + ".mp3")
            
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )
            1==1

            trackWithAudio.append(compiledTrack)
    return trackWithAudio

tracks = getTracks()
1 == 1

            





