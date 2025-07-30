from bridge import Bridge
import time
import numpy as np
from sampler import *
import os
import struct
import matplotlib.pyplot as plt
import argparse
import pandas as pd

SAMPLING_RATE = 1000
WIN = 1000  # window length
NFFT = 128
NPERSEG = 128
NOVERLAP = 64

model_spec = tf.keras.models.load_model("training/models/spec/first_9822_accuracy.keras")
model_lstm = tf.keras.models.load_model("training/models/lstm/mlp_stats_raina.keras")

print("Model loaded successfully.")

receiver = Bridge("0.0.0.0", 12345, recieve=True)
receiver.connect()

sender = Bridge("172.20.10.45", 12345)
sender = Bridge("172.20.10.45", 12345)
sender.connect()

count = 0

# ch1, ch2 = [1.5,]*1000, [1.5,]*1000
def subtract_noise(data, noise_profile):
    """
    Subtracts the noise profile from the data.
    """
    if noise_profile is not None:
        noise_profile = noise_profile[:, np.newaxis, :]
        print(noise_profile.shape)
        print("Noise profile mean:", np.mean(noise_profile))
        print("Spec mean:", np.mean(data))
        data = np.clip(data - noise_profile, a_min=0, a_max=None)
        print("Spec after noise profile subtraction mean:", np.mean(data))

    return data


def predict_spectrogram(clean1, clean2, noise_profile=None):

    # Combine and convert to spectrogram
    spec1 = get_spectrogram(tf.convert_to_tensor(clean1, dtype=tf.float32))
    spec2 = get_spectrogram(tf.convert_to_tensor(clean2, dtype=tf.float32))    # shape (freq, time, channels)

    spec = np.stack([spec1, spec2], axis=-1) 

    spec = np.expand_dims(spec, axis=0)

    if noise_profile is not None:
        spec = subtract_noise(spec, noise_profile)

    print(spec1.shape, spec2.shape, spec.shape)
    # shape (1, freq, time, channels)

    # Predict using
    preds = model_spec.predict(spec)
    label = np.argmax(preds, axis=1)
    return spec, label
    # print(preds)

    # print(f"Predicted label: {label}")

def predict_LSTM(clean1, clean2):

    X = np.stack([clean1, clean2], axis=0)

    means = np.mean(X, axis=1)
    stds = np.std(X, axis=1)
    vars_ = np.var(X, axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))
    sum_abs_diff = np.sum(np.abs(np.diff(X, axis=1)), axis=1)

    feat = np.concatenate([means, stds, vars_, rms, sum_abs_diff], axis=0).reshape(1, -1)
    preds = model_lstm.predict(feat, verbose=0)
    print(preds)
    pred = np.argmax(preds, axis=1)
    return pred

def collect_noise_profile():
    n_samples = 2000
    profile_ch1 = np.array([])
    profile_ch2 = np.array([])
    i = 0
    while len(profile_ch1) < n_samples or len(profile_ch2) < n_samples:
        data = receiver.receive_data()
        values = struct.unpack('200H', data)

        v1 = np.array(values[0::2])
        v2 = np.array(values[1::2])

        v1 = np.round(v1 * 3.3 / 65535, 6)
        v2 = np.round(v2 * 3.3 / 65535, 6)

        nch1 = np.array(v1)
        nch2 = np.array(v2)

        profile_ch1 = np.concatenate([nch1, profile_ch1])
        profile_ch2 = np.concatenate([nch2, profile_ch2])

        i+= len(v1)     

        print(f"Collected {i+1}/{n_samples} noise samples")

    spec1 = get_spectrogram(tf.convert_to_tensor(profile_ch1, dtype=tf.float32))
    spec2 = get_spectrogram(tf.convert_to_tensor(profile_ch2, dtype=tf.float32))
    print("Noise sample shape:", spec1.shape, spec2.shape)

    noise_spec = np.stack([spec1, spec2], axis=-1)

    noise_profile = np.mean(noise_spec, axis=1) # time average 

    return noise_profile

def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--save_file', dest='array_name', required=False)
    # group.add_argument('--no_run', dest='no_run', required=False, action='store_true', default=False)

    args = parser.parse_args()

    total_set = np.array([])
    
    ch1 = np.array([])
    ch2 = np.array([])

    # print("Collecting noise samples...")
    # time.sleep(1)

    # profile = collect_noise_profile()
    # print("Noise profile shape: ", profile.shape)

    while True:
        try:
            # i = input("Enter data to send: ")
            data = receiver.receive_data()

            values = struct.unpack('200H', data)

            v1 = np.array(values[0::2])
            v2 = np.array(values[1::2])

            v1 = np.round(v1 * 3.3 / 65535, 6)
            v2 = np.round(v2 * 3.3 / 65535, 6)

            nch1 = np.array(v1)
            nch2 = np.array(v2)
            # print(ch1, ch2)
            nch1 = cleanup(nch1)
            nch2 = cleanup(nch2)

            ch1 = np.concatenate([ch1, nch1])[-WIN:]
            ch2 = np.concatenate([ch2, nch2])[-WIN:]

            print(np.mean(ch1), np.mean(ch2))

            if len(ch1) < WIN or len(ch2) < WIN:
                # print("Not enough data to process.")
                continue

            spec, pred = predict_spectrogram(ch1, ch2)
            print("spectrogram Prediction:", pred)
            
            total_set = np.append(total_set, spec)
            # ch1, ch2 = [], []

        except KeyboardInterrupt:
            print("Exiting...")
            f1, t1, spec1 = stft(tf.convert_to_tensor(ch1, dtype=tf.float32), sampling_rate=SAMPLING_RATE, return_full=True)
            f2, t2, spec2 = stft(tf.convert_to_tensor(ch2, dtype=tf.float32), sampling_rate=SAMPLING_RATE, return_full=True)

            spec = np.stack([spec1, spec2], axis=-1)
            # spec = np.expand_dims(spec, axis=0)
            # spec = subtract_noise(spec, profile)

            spec_no_batch = spec[0]        # shape (129, 9, 2)

            spec1 = spec_no_batch[..., 0]  # (129, 9)
            spec2 = spec_no_batch[..., 1] 

            print(spec.shape)
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

            pcm1 = axes[0].pcolormesh(t1, f1, spec1, shading='gouraud', cmap='viridis')
            axes[0].set_title('Spectrogram - Channel 1')
            axes[0].set_ylabel('Frequency [Hz]')
            fig.colorbar(pcm1, ax=axes[0], label='Amplitude')

            pcm2 = axes[1].pcolormesh(t2, f2, spec2, shading='gouraud', cmap='viridis')
            axes[1].set_title('Spectrogram - Channel 2')
            axes[1].set_ylabel('Frequency [Hz]')
            axes[1].set_xlabel('Time [sec]')
            fig.colorbar(pcm2, ax=axes[1], label='Amplitude')

            plt.tight_layout()
            plt.savefig('spec.png')
            plt.close()

            if args.array_name:
                fname = args.array_name
                print(total_set.shape)
                np.save(fname, total_set)
            receiver.close()
            sender.close()
            break

if __name__ == "__main__":
    main()