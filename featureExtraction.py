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
    """
    Retrieves the file path for a given track ID from the database.

    Parameters:
    - track_id (int): The ID of the track to retrieve.

    Returns:
    - str: The file path of the track, formatted for the current operating system.
    """
    query = "SELECT * FROM tracks WHERE track_id = %s"
    values = (track_id,)
    cursor.execute(query, values)

    track = cursor.fetchone()

    if os.name == 'posix':
        return track[2].replace("\\", "/")
    else:
        return track[2]


def get_rhythm_info(track_id):
    """
    Extracts tempo and beat information from an audio file specified by its track ID.

    Parameters:
    - track_id (int): The ID of the track.

    Returns:
    - tuple: A tuple containing tempo as a tensor and beats as a tensor.
    """
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
    """
    Generates a Mel spectrogram from an audio file specified by its track ID.

    Parameters:
    - track_id (int): The ID of the track.

    Returns:
    - torch.Tensor: The Mel spectrogram tensor.
    """
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
    """
    Generates MFCCs (Mel Frequency Cepstral Coefficients) from an audio file specified by its track ID.

    Parameters:
    - track_id (int): The ID of the track.

    Returns:
    - torch.Tensor: The MFCC tensor.
    """
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
    """
    Generates a Mel spectrogram from an audio file specified by its path.

    Parameters:
    - track_id (int): The ID of the track.

    Returns:
    - torch.Tensor: The Mel spectrogram tensor.
    """
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
    spec_plot = spec
    spec_noramlised = torch.log(spec + 1e-6)
    return spec_plot, spec_noramlised, str(file_id)

def gen_mffc_path(file_path):
    """
    Generates MFCCs (Mel Frequency Cepstral Coefficients) from an audio file specified by its path.

    Parameters:
    - track_id (int): The ID of the track.

    Returns:
    - torch.Tensor: The MFCC tensor.
    """
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
    """
    Extracts tempo and beat information from an audio file specified by its path.

    Parameters:
    - track_id (int): The ID of the track.

    Returns:
    - tuple: A tuple containing tempo as a tensor and beats as a tensor.
    """
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
    """
    Plots and saves the waveform of an audio track. (not used)

    Parameters:
    - track_id (int): The ID of the track.
    - waveform (torch.Tensor): The waveform tensor.
    - sr (int): Sample rate of the waveform.
    - batchId (str): Identifier for the batch, used to organize saved plots.

    Returns:
    - None: The plot is saved to a file and not returned.
    """
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
    plt.close()

def plot_spectrogram(track_id, specgram):
    """
    Plots and saves Mel spectrogram of an audio track.

    Parameters:
    - track_id (int): The ID of the track.
    - waveform (torch.Tensor): The Spectrogram tensor.

    Returns:
    - None: The plot is saved to a file and not returned.
    """
    if not isinstance(specgram, np.ndarray):
        specgram = np.array(specgram)
    combined_specgram = np.mean(specgram, axis=0)
    plt.figure(figsize=(10, 6))
    img = librosa.display.specshow(librosa.power_to_db(combined_specgram, ref=np.max),
                                   y_axis='mel', fmax=8000, x_axis='time', sr=44100)
    plt.tight_layout()
    plt.xlabel("Time (Min:sec)", color='black')
    plt.ylabel("Frequency (Hz)", color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')
    plt.gca().set_facecolor('none')
    plt.colorbar(img, format='%+2.0f dB', label='Intensity (dB)')
    plt.savefig("static/images/" + str(track_id) + ".png", transparent=True)
    plt.close()
