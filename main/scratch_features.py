import numpy as np
from scipy import signal
from scipy.signal import periodogram
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.decomposition import PCA
import seaborn as sns
import joblib

SAMPLING_RATE = 1000          
WINDOW_SEC = 1.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLING_RATE)
OVERLAP = 0.5
STEP = int(WINDOW_SAMPLES * (1 - OVERLAP))
RANDOM_STATE = 42
pca = joblib.load("/home/nine/ecz-ware/main/models/pca_model_o.joblib")
scaler = joblib.load("/home/nine/ecz-ware/main/models/scaler_o.joblib")

def sliding_window(arr, win_size, step):
    """Generator of windows (start_idx, window_array)."""
    n = arr.shape[0]
    for start in range(0, n - win_size + 1, step):
        yield start, arr[start:start + win_size]

# Time-domain features commonly used in HAR/EMG scratch detection
def time_domain_features(win):
    # win: 1D numpy array
    mean = np.mean(win)
    std = np.std(win)
    mav = np.mean(np.abs(win))
    rms = np.sqrt(np.mean(win**2))
    wl = np.sum(np.abs(np.diff(win)))              # waveform length
    zc = ((win[:-1] * win[1:]) < 0).sum()         # zero crossings
    # simple slope sign changes
    ssc = np.sum(((np.diff(win[:-1]) * np.diff(win[1:])) < 0).astype(int))
    # min/max
    mn = win.min(); mx = win.max()
    return [mean, std, mav, rms, wl, zc, ssc, mn, mx]

def stft_band_energy(win, fs=SAMPLING_RATE, nperseg=256, noverlap=128, bands=None):
    """
    Compute STFT magnitude, then summarize band energies.
    bands: list of (low, high) Hz bands to summarize. if None, default bands used.
    returns list of band energies (log-scaled) and global spectral entropy & dominant freq.
    """
    if bands is None:
        bands = [(0,5), (5,20), (20,50), (50,120), (120,300)]  # example bands (tweakable)
    f, t, S = signal.stft(win, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=256, boundary='zeros')
    S_mag = np.abs(S)  # shape (freq_bins, time_frames)
    psd = np.mean(S_mag**2, axis=1) + 1e-12   # power per freq bin
    band_energies = []
    for (lo, hi) in bands:
        idx = np.where((f >= lo) & (f < hi))[0]
        if idx.size == 0:
            band_energies.append(0.0)
        else:
            band_energies.append(np.log10(np.sum(psd[idx]) + 1e-12))
    # spectral entropy
    p_norm = psd / psd.sum()
    spec_entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12))
    # dominant frequency
    dom_freq = f[np.argmax(psd)]
    return band_energies + [spec_entropy, dom_freq]

def scale_features(_X):

    X = _X.copy()

    X[:, 0] = X[:, 0] * 3.3 / 65535.0
    X[:, 1] = X[:, 1] * 3.3 / 65535.0

    X[:, 2] = (X[:, 2] - 32768) / 16384.0
    X[:, 3] = (X[:, 3] - 32768) / 16384.0
    X[:, 4] = (X[:, 4] - 32768) / 16384.0

    return X

def get_model_features(_X):

    X = _X.copy()

    X = scale_features(X)
    feats = []

    feats += time_domain_features(X[:, 0])
    feats += time_domain_features(X[:, 1])

    mag = np.sqrt(X[:, 2]**2 + X[:, 3]**2 + X[:, 4]**2)

    feats += time_domain_features(mag)
    feats += stft_band_energy(mag)

    feats = np.array(feats).reshape(1, -1)

    feats = scaler.transform(feats)
    feats = pca.transform(feats)

    return np.array(feats)