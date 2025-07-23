from bridge import Bridge
import time
import numpy as np
from sampler import *
import os
import struct
import matplotlib.pyplot as plt

SAMPLING_RATE = 1000
WIN = 1000  # window length
NFFT = 128
NPERSEG = 128
NOVERLAP = 64

model = tf.keras.models.load_model("training/models/first_9921_accuracy.keras")

print("Model loaded successfully.")

receiver = Bridge("0.0.0.0", 12345, recieve=True)
receiver.connect()

sender = Bridge("172.20.10.45", 12345)
sender = Bridge("172.20.10.45", 12345)
sender.connect()

count = 0

# ch1, ch2 = [1.5,]*1000, [1.5,]*1000

ch1 = np.zeros(SAMPLING_RATE)
ch2 = np.zeros(SAMPLING_RATE)

while True:
    try:
        # i = input("Enter data to send: ")
        data = receiver.receive_data()

        values = struct.unpack('200H', data)

        v1 = np.array(values[0::2])
        v2 = np.array(values[1::2])

        v1 = v1 * 3.3 / 65535
        v2 = v2 * 3.3 / 65535
    
        nch1 = np.array(v1)
        nch2 = np.array(v2)
        # print(ch1, ch2)

        ch1 = np.roll(ch1, -len(nch1))
        ch2 = np.roll(ch2, -len(nch2))
        ch1[-len(v1):] = nch1
        ch2[-len(v2):] = nch2

        print(ch1.shape, ch2.shape)

        # Cleanup with fs=1000
        ch1 = cleanup(ch1)
        ch2 = cleanup(ch2)

        # Combine and convert to spectrogram
        spec1 = get_spectrogram(tf.convert_to_tensor(ch1, dtype=tf.float32))
        spec2 = get_spectrogram(tf.convert_to_tensor(ch2, dtype=tf.float32))    # shape (freq, time, channels)

        spec = np.stack([spec1, spec2], axis=-1) 

        spec = np.expand_dims(spec, axis=0)

        print(spec1.shape, spec2.shape, spec.shape)
        # shape (1, freq, time, channels)

        # Predict
        preds = model.predict(spec)
        label = np.argmax(preds, axis=1)

        # print(preds)

        print(f"Predicted label: {label}")
        # ch1, ch2 = [], []

    except KeyboardInterrupt:
        print("Exiting...")
        f1, t1, spec1 = stft(tf.convert_to_tensor(ch1, dtype=tf.float32), sampling_rate=SAMPLING_RATE, return_full=True)
        f2, t2, spec2 = stft(tf.convert_to_tensor(ch2, dtype=tf.float32), sampling_rate=SAMPLING_RATE, return_full=True)
        print("Channel 1 mean amplitude:", np.mean(spec1))
        print("Channel 2 mean amplitude:", np.mean(spec2))
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
        plt.savefig('rest_data_both_channels.png')
        plt.close()
        receiver.close()
        sender.close()
        break