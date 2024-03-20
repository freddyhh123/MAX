# Standard Python libs
import os
import uuid
# Third-party libs
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Setting the 'Agg' backend for matplotlib
import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment
# PyTorch libs
import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.transforms import MFCC
# Local libs
from databaseConfig import connect

db = connect()
cursor = cursor = db.cursor()

def get_file(track_id):
    query = "SELECT * FROM tracks WHERE track_id = %s"
    values = (track_id,)
    cursor.execute(query, values)

    track = cursor.fetchone()

    if os.name == 'posix':
        return track[2].replace("\\", "/")
    else:
        return track[2]


def get_rhythm_info(track_id):
    track_path = get_file(track_id)

    wav, sample_rate = torchaudio.load(track_path, normalize = True)
    resample_rate = 44100
    resampler = T.Resample(sample_rate, resample_rate, dtype=wav.dtype)
    wav = resampler(wav)
    sample_rate = 44100

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0).numpy()
    else:
        wav = wav.squeeze().numpy()

    tempo, beats = librosa.beat.beat_track(y=wav,sr=sample_rate)
    tempo = torch.tensor(tempo)
    beats = torch.tensor(beats)

    return tempo, beats

    

def gen_spectrogram(track_id):
    track_path = get_file(track_id)

    wav, sample_rate = torchaudio.load(track_path, normalize = True)

    resample_rate = 44100
    resampler = T.Resample(sample_rate, resample_rate, dtype=wav.dtype)
    wav = resampler(wav)
    sample_rate = 44100

    transform = T.MelSpectrogram(sample_rate,n_mels = 128, n_fft = 2048,hop_length = 512)
    spec = transform(wav)

    spec = torch.log(spec + 1e-6)

    return spec

def gen_mfcc(track_id):
    query = "SELECT * FROM tracks WHERE track_id = %s"
    values = (track_id,)
    cursor.execute(query, values)

    track = cursor.fetchone()

    if os.name == 'posix':
        track_path = track[2].replace("\\", "/")
    else:
        track_path = track[2]

    wav, sample_rate = torchaudio.load(track_path, normalize = True)

    resample_rate = 44100
    resampler = T.Resample(sample_rate, resample_rate, dtype=wav.dtype)
    wav = resampler(wav)
    sample_rate = 44100

    mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=13, melkwargs={'hop_length': 512, 'n_fft': 2048})
    mfccs = mfcc_transform(wav)

    return mfccs

def gen_spectrogram_path(file_path):
    file_id = uuid.uuid4()
    if os.name == 'posix':
        file_path = file_path.replace("\\", "/")

    wav, sample_rate = torchaudio.load(file_path, normalize = True)

    resample_rate = 44100
    resampler = T.Resample(sample_rate, resample_rate, dtype=wav.dtype)
    wav = resampler(wav)
    sample_rate = 44100

    transform = T.MelSpectrogram(sample_rate,n_mels = 128, n_fft = 2048,hop_length = 512)
    spec = transform(wav)
    plot_spectrogram(file_id, spec)
    spec_noramlised = torch.log(spec + 1e-6)
    return spec_noramlised, str(file_id)

def gen_mffc_path(file_path):
    if os.name == 'posix':
        file_path = file_path.replace("\\", "/")
    wav, sample_rate = torchaudio.load(file_path, normalize = True)

    resample_rate = 44100
    resampler = T.Resample(sample_rate, resample_rate, dtype=wav.dtype)
    wav = resampler(wav)
    sample_rate = 44100

    mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=13, melkwargs={'hop_length': 512, 'n_fft': 2048})
    mfccs = mfcc_transform(wav)

    return mfccs

def get_rhythm_info_path(file_path):

    file_id = uuid.uuid4()
    if os.name == 'posix':
        file_path = file_path.replace("\\", "/")

    wav, sample_rate = torchaudio.load(file_path, normalize = True)
    resample_rate = 44100
    resampler = T.Resample(sample_rate, resample_rate, dtype=wav.dtype)
    wav = resampler(wav)
    sample_rate = 44100

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0).numpy()
    else:
        wav = wav.squeeze().numpy()

    tempo, beats = librosa.beat.beat_track(y=wav,sr=sample_rate)
    tempo = torch.tensor(tempo)
    beats = torch.tensor(beats)

    return tempo, beats


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


def plot_spectrogram(track_id, specgram):
    if not isinstance(specgram, np.ndarray):
        specgram = np.array(specgram)
    combined_specgram = np.mean(specgram, axis=0)
    plt.figure(figsize=(10, 6), facecolor='none', edgecolor='none')
    librosa.display.specshow(librosa.power_to_db(combined_specgram, ref=np.max),
                             y_axis='mel', fmax=8000, x_axis='time', sr=44100)
    plt.tight_layout()
    plt.xlabel("Time",color='black')
    plt.ylabel("Hz", color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')
    plt.gca().set_facecolor('none')
    plt.savefig("static/images/" + str(track_id) + ".png", transparent=True)
