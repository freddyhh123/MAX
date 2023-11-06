import spotipy
import pandas as pd
import requests
from io import BytesIO
from spotipy.oauth2 import SpotifyClientCredentials

from pydub import AudioSegment
import matplotlib.pyplot as pyplt
from scipy.io import wavfile
from scipy import signal
from tempfile import mktemp



def getTracks(sp):
    cid = "516f5defb70e4e0994acea4cc99d1d4c"
    secret = "0038b1fca65d467dbd0d2e1437041545"

    client_credentials_manager = SpotifyClientCredentials(
    client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    playlist_uri = 'spotify:playlist:44laAM2s0CKDb9AdBpKtfZ'

    results = sp.playlist_items(playlist_uri)
    trackWithAudio = []

    for entry in results['items'][:100]:
        track = entry["track"]
        if track['preview_url']:
            track['genres'] = []
            track['features'] = sp.audio_features(track["id"])[0]
            for artist in track['artists']:
                track["genres"] = sp.artist(artist["id"])["genres"]
            trackWithAudio.append(entry)
    return trackWithAudio

            





