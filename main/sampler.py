import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, periodogram

SAMPLING_RATE = 1000
WIN = 1000  # window length
NFFT = 256
NPERSEG = 255
NOVERLAP = 124

def normalize(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-8)


def bandpass_filter(data, lowcut=5.0, highcut=90.0, fs=SAMPLING_RATE, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def find_noise_frequency(data, fs=SAMPLING_RATE):
    f, Pxx = periodogram(data, fs=fs)
    idx = np.argmax(Pxx)
    return f[idx]

def adaptive_notch_filter(data, fs=SAMPLING_RATE, quality=30):
    freq = find_noise_frequency(data, fs=fs)
    nyq = 0.5 * fs
    b, a = iirnotch(freq/nyq, quality)
    return filtfilt(b, a, data)

def cleanup(data):
    data = normalize(data)
    data = bandpass_filter(data)
    data = adaptive_notch_filter(data)
    return data

def stft(x, sampling_rate=SAMPLING_RATE, return_full=False):
    f, t, spec = signal.stft(x.numpy(), fs=sampling_rate, nperseg=NPERSEG, noverlap = NOVERLAP, nfft=NFFT, boundary='zeros')
    if return_full:
        return f, t, tf.convert_to_tensor(np.abs(spec))
    else:
        return tf.convert_to_tensor(np.abs(spec))

def get_spectrogram(data: np.ndarray, return_full=False):
    # f, t, z = signal.stft(data, fs=200, nperseg=128, noverlap=50, nfft=128)
    spectrogram = tf.py_function(func=stft, inp=[data], Tout=tf.float32)
    spectrogram = tf.image.resize(spectrogram[..., tf.newaxis], [129, 124])
    spectrogram = tf.squeeze(spectrogram, -1)
    # spectrogram.set_shape((129, 124))
    return spectrogram