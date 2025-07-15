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

sender = Bridge("172.20.10.22", 12345)
sender.connect()

count = 0

# ch1, ch2 = [1.5,]*1000, [1.5,]*1000

ch1, ch2 = [], []

while True:
    try:
        # i = input("Enter data to send: ")
        line = receiver.receive_data()
        if line[0:3] == "EMG":
            sender.send_data("True")  # signal back to Pico to start sending
            
            # Initialize buffers
            
            if len(ch1) < SAMPLING_RATE + 1:
                
                if line.startswith("EMG") and "#" in line:
                    try:
                        parts = line.strip().split("#")
                        v1 = float(parts[0][3:])
                        v2 = float(parts[1])
                        ch1.append(v1)
                        ch2.append(v2)
                        count += 1
                        print(count)
                    except ValueError:
                        continue

                
            else:
                
                ch1 = ch1[1:]
                ch2 = ch2[1:]
            
                nch1 = np.array(ch1)
                nch2 = np.array(ch2)
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

            

        else:
            receiver.connect()
            time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting...")
        receiver.close()
        sender.close()
        break