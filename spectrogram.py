import torch
import torchaudio

import numpy as np

import torchaudio.transforms as T
import matplotlib.pyplot as plt

import urllib.request

import librosa

import os

def gen_spectrogram(compiledTrack):
    if not os.path.isdir("audio/"+compiledTrack["batchId"]):
        os.mkdir("audio/"+compiledTrack['batchId'])

    urllib.request.urlretrieve(compiledTrack['preview_url'], "audio/"+ compiledTrack['batchId'] +"/"+ str(compiledTrack['id']) + ".mp3")

    wav, sample_rate = torchaudio.load("audio/"+compiledTrack["batchId"] +"/"+ str(compiledTrack['id']) + ".mp3", normalize = True)

    resample_rate = 16000
    resampler = T.Resample(sample_rate, resample_rate, dtype=wav.dtype)
    wav = resampler(wav)
    sample_rate = 16000

    transform = T.MelSpectrogram(sample_rate,n_mels=64)
    spec = transform(wav)

    compiledTrack["wav"] = wav
    compiledTrack["spectrogram"] = spec


    plot_waveform(compiledTrack['id'], wav, sample_rate, compiledTrack["batchId"])
    plot_spectrogram(compiledTrack['id'], spec[0], compiledTrack["batchId"])

    return compiledTrack



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
    
