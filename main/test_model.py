from bridge import Bridge
import time
import numpy as np
from sampler import *
import os

SAMPLING_RATE = 1000
WIN = 1000  # window length
NFFT = 128
NPERSEG = 128
NOVERLAP = 64

model = tf.keras.models.load_model("training/models/first_9412_accuracy.keras")

print("Model loaded successfully.")

receiver = Bridge("0.0.0.0", 12345, recieve=True)
receiver.connect()

sender = Bridge("172.20.10.45", 12345)
sender.connect()

count = 0

# ch1, ch2 = [1.5,]*1000, [1.5,]*1000

ch1, ch2 = [], []

while True:
    try:
        # i = input("Enter data to send: ")
        line = receiver.receive_data()

        values = struct.unpack('2000H', data)

        v1 = np.array(values[0::2])
        v2 = np.array(values[1::2])

        v1 = v1 * 3.3 / 65535
        v2 = v2 * 3.3 / 65535
    
        nch1 = np.array(v1)
        nch2 = np.array(v2)
        # print(ch1, ch2)

        # Cleanup with fs=1000
        nch1 = cleanup(ch1)
        nch2 = cleanup(ch2)

        # Combine and convert to spectrogram
        current = np.stack([ch1, ch2], axis=1)  # shape (1000, 2)
        spec = make_spectrogram(current)        # shape (freq, time, channels)

        # shape (1, freq, time, channels)

        # Predict
        preds = model.predict(spec)
        label = np.argmax(preds, axis=1)

        # print(preds)

        print(f"Predicted label: {label}")
        # ch1, ch2 = [], []

    except KeyboardInterrupt:
        print("Exiting...")
        receiver.close()
        sender.close()
        break