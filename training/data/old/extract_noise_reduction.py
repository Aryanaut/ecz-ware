# extract.py
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, periodogram
import matplotlib.pyplot as plt

def bandpass_filter(data, lowcut=20.0, highcut=90.0, fs=200.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def adaptive_notch_filter(data, freq, fs=200.0, quality=30):
    nyq = 0.5 * fs
    b, a = iirnotch(freq/nyq, quality)
    return filtfilt(b, a, data)

def find_noise_frequency(data, fs=200.0):
    f, Pxx = periodogram(data, fs=fs)
    idx = np.argmax(Pxx)
    return f[idx]

def extract_and_plot(ts, window_size=200, step=100, fs=200.0):
    os.makedirs('features', exist_ok=True)
    df = pd.read_csv(f"data/emg_{ts}.csv")
    v1_raw, v2_raw, labels = df['v1'].values, df['v2'].values, df['label'].values

    # find noise freq on v1 (both sensors usually same)
    noise_freq = find_noise_frequency(v1_raw)
    print(f"identified dominant noise freq: {noise_freq:.1f} Hz")

    # filter + remove noise
    v1 = bandpass_filter(v1_raw, fs=fs)
    v2 = bandpass_filter(v2_raw, fs=fs)
    v1_filt = adaptive_notch_filter(v1, noise_freq, fs=fs)
    v2_filt = adaptive_notch_filter(v2, noise_freq, fs=fs)

    # extract features
    X, y = [], []
    for start in range(0, len(v1_filt)-window_size+1, step):
        w1, w2 = v1_filt[start:start+window_size], v2_filt[start:start+window_size]
        label = np.bincount(labels[start:start+window_size]).argmax()
        feat = []
        for w in [w1, w2]:
            feat += [
                np.mean(w), np.std(w), np.var(w),
                np.sqrt(np.mean(np.square(w))),
                np.sum(np.abs(np.diff(w)))
            ]
        X.append(feat)
        y.append(label)

    X, y = np.array(X), np.array(y)
    out_file = f"features/feat_{ts}.npz"
    np.savez(out_file, X=X, y=y, noise_freq=noise_freq)
    print(f"saved features: {out_file} shape={X.shape}")

    # plot raw vs filtered
    fig, axs = plt.subplots(2,1, figsize=(12,6))
    axs[0].plot(v1_raw, alpha=0.5, label='raw v1')
    axs[0].plot(v1_filt, label='filtered v1')
    axs[0].legend(); axs[0].set_title('sensor 1')
    axs[1].plot(v2_raw, alpha=0.5, label='raw v2')
    axs[1].plot(v2_filt, label='filtered v2')
    axs[1].legend(); axs[1].set_title('sensor 2')
    plt.tight_layout()
    plot_file = f"features/plot_{ts}.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"saved plot: {plot_file}")

if __name__ == "__main__":
    ts = input("enter timestamp of raw data (e.g., 1720...): ")
    extract_and_plot(ts)
