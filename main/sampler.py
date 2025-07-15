import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, periodogram

SAMPLING_RATE = 1000
WIN = 1000  # window length
NFFT = 128
NPERSEG = 128
NOVERLAP = 64

def normalize(data):
    data = data - np.mean(data)
    data = (data - data.min()) / (data.max() - data.min())
    return data

def bandpass_filter(data, lowcut=20.0, highcut=90.0, fs=200.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def find_noise_frequency(data, fs=200.0):
    f, Pxx = periodogram(data, fs=fs)
    idx = np.argmax(Pxx)
    return f[idx]

def adaptive_notch_filter(data, fs=200.0, quality=30):
    freq = find_noise_frequency(data, fs=fs)
    nyq = 0.5 * fs
    b, a = iirnotch(freq/nyq, quality)
    return filtfilt(b, a, data)

def cleanup(data):
    data = normalize(data)
    data = bandpass_filter(data)
    data = adaptive_notch_filter(data)
    return data

def stft(x, sampling_rate=1000):
    f, t, spec = signal.stft(x.numpy(), fs=sampling_rate, nperseg=128, noverlap = 64, nfft=128, boundary='zeros')
    return tf.convert_to_tensor(np.abs(spec))

def make_spectrogram(win):
    specs = []
    for i in range(2):  # two channels
        f, t, Z = signal.stft(win[:, i], fs=SAMPLING_RATE,
                              nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
        specs.append(np.abs(Z))
    spec = np.stack(specs, axis=0)  # shape (2, freq_bins, time_steps)
    # Add batch dim
    return np.expand_dims(spec, axis=0).astype(np.float32)