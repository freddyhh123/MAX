import torch
import torchaudio

import numpy as np

import torchaudio.transforms as T
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import urllib.request

import librosa

import os

from databaseConfig import connect

db = connect()
cursor = cursor = db.cursor()


def gen_spectrogram(track_id):
    query = "SELECT * FROM tracks WHERE track_id = %s"
    values = (track_id,)
    cursor.execute(query, values)

    track = cursor.fetchone()

    wav, sample_rate = torchaudio.load(track[2], normalize = True)

    resample_rate = 44100
    resampler = T.Resample(sample_rate, resample_rate, dtype=wav.dtype)
    wav = resampler(wav)
    sample_rate = 44100

    transform = T.MelSpectrogram(sample_rate,n_mels = 128, n_fft = 2048)
    spec = transform(wav)


    #plot_waveform(track[0], wav, sample_rate, track[5])
    #plot_spectrogram(track[0], spec[0], track[5])
    return spec

def gen_central_spectroid(track_id):
    query = "SELECT * FROM tracks WHERE track_id = %s"
    values = (track_id,)
    cursor.execute(query, values)
    track = cursor.fetchone()
    wav, sample_rate = librosa.load(track[2])
    resample_rate = 44100
    wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=resample_rate)
    sample_rate = 44100
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sample_rate, n_fft=2048, hop_length= (2048//4))
    centroid = centroid[:, :2580]

    return torch.tensor(centroid)



def plot_waveform(track_id, waveform, sr,batchId):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr
    plt.figure()
    plt.plot(time_axis, waveform[0], linewidth=1)
    plt.grid(True)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    if not os.path.isdir("images/"+batchId):
        os.mkdir("images/"+batchId)
    if not os.path.isdir("images/"+batchId+"/wav"):
        os.mkdir("images/"+batchId+"/wav/")
    plt.savefig("images/"+batchId+"/wav/" + track_id + ".png", transparent=True)


def plot_spectrogram(track_id, specgram,batchId):
    plt.figure()
    librosa.display.specshow(librosa.power_to_db(specgram))
    plt.colorbar()
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    if not os.path.isdir("images/"+batchId):
        os.mkdir("images/"+batchId)
    if not os.path.isdir("images/"+batchId+"/spec"):
        os.mkdir("images/"+batchId+"/spec/")
    plt.savefig("images/"+batchId+"/spec/" + track_id + ".png", transparent=True)
    
