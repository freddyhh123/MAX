import torch
import torchaudio

import torchaudio.transforms as T
import matplotlib.pyplot as plt

import urllib.request

import librosa

def gen_spectrogram(compiledTrack):

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

    return compiledTrack



def plot_waveform(track_id, waveform, sr):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr
    plt.figure()
    plt.plot(time_axis, waveform[0], linewidth=1)
    plt.grid(True)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.savefig("images/wav/" + track_id + ".png", transparent=True)


def plot_spectrogram(track_id, specgram):
    plt.figure()
    librosa.display.specshow(librosa.power_to_db(specgram))
    plt.colorbar()
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.savefig("images/spec/" + track_id + ".png", transparent=True)