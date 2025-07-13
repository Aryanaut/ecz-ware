# extract.py
import pandas as pd
import numpy as np
import os

def extract_features(ts, window_size=200, step=100):
    os.makedirs('features', exist_ok=True)
    df = pd.read_csv(f"data/emg_{ts}.csv")
    v1s, v2s, labels = df['v1'].values, df['v2'].values, df['label'].values
    X, y = [], []

    for start in range(0, len(v1s)-window_size+1, step):
        w1, w2 = v1s[start:start+window_size], v2s[start:start+window_size]
        label = np.bincount(labels[start:start+window_size]).argmax()
        feat = []

        for w in [w1, w2]:
            feat += [
                np.mean(w), np.std(w), np.var(w), 
                np.sqrt(np.mean(np.square(w))),             # RMS
                np.sum(np.abs(np.diff(w)))                  # waveform length
            ]
        X.append(feat)
        y.append(label)
    out = f"features/feat_{ts}.npz"
    np.savez(out, X=np.array(X), y=np.array(y))
    print(f"saved {out} shape: {np.array(X).shape}")

if __name__ == "__main__":
    ts = input("timestamp (e.g., 1720000000): ")
    extract_features(ts)
