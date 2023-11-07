import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials

import torch
import torchaudio

import urllib.request
from pydub import AudioSegment

import torchaudio.transforms as T
import matplotlib.pyplot as plt

import librosa

def getTracks():
    cid = "516f5defb70e4e0994acea4cc99d1d4c"
    secret = "0038b1fca65d467dbd0d2e1437041545"

    client_credentials_manager = SpotifyClientCredentials(
    client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout=10, retries=10)
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
            
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )
            compiledTrack["mel_spectrogram"] = mel_spectrogram

            wav, sample_rate = torchaudio.load("audio/" + str(compiledTrack['id']) + ".mp3")
            spectrogram = T.Spectrogram(n_fft=512)
            spec = spectrogram(wav)

            compiledTrack["wav"] = wav
            compiledTrack["spectrogram"] = spec


            plot_waveform(compiledTrack['id'], wav, sample_rate)
            plot_spectrogram(compiledTrack['id'], spec[0])
            
            trackWithAudio.append(compiledTrack)
    return trackWithAudio


def plot_waveform(track_id, waveform, sr):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    plt.plot(time_axis, waveform[0], linewidth=1)
    plt.grid(True)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.savefig("images/wav/" + track_id + ".png", transparent=True)


def plot_spectrogram(track_id, specgram):
    plt.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.savefig("images/spec/" + track_id + ".png", transparent=True)


tracks = getTracks()
1 == 1

            





