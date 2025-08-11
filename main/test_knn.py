from bridge import Bridge
import time
import numpy as np
import socket, struct
from scratch_features import *
import numpy as np
import tensorflow as tf
import joblib

sample_rate = 1000
batch_size = 100
values_per_sample = 5
expected_bytes = batch_size * values_per_sample * 2  # 2 bytes per unsigned short

model_name = '/home/nine/ecz-ware/main/models/knn_model_2_o.joblib'

def init_features(reciever, n_samples):
    data = np.empty((0, 5))
    for i in range(n_samples // batch_size):
        packet = reciever.receive_data()
        if len(packet) != expected_bytes:
            print(f"Warning: received {len(packet)} bytes, expected {expected_bytes}. Skipping.")
            continue

        values = struct.unpack('500H', packet)
        values = np.array(values)
        values = values.reshape(batch_size, values_per_sample)

        if len(data) % 100 == 0:
            print(f"Received {len(data)} samples, shape: {data.shape}")
        value_stack = values

        if data.size == 0:
            data = value_stack
        else:
            data = np.vstack((data, value_stack))

    return data

def main():
    s_count = 0

    model = joblib.load(model_name)
    print("Model: ", model_name, "loaded successfully.")

    fname = "test.npy"

    reciever = Bridge("0.0.0.0", 12345, recieve=True)
    reciever.connect()

    sender = Bridge("172.20.10.45", 12345)
    sender.connect()

    n_samples = 1000
    state = np.zeros((n_samples, values_per_sample), dtype=np.float32)
    print("Initial state shape:", state.shape)

    while True:
        try:
            # print("I'm trying man")a
            # i = input("Enter data to send: ")
            data = reciever.receive_data()

            if len(data) != expected_bytes:
                print(f"Warning: received {len(data)} bytes, expected {expected_bytes}. Skipping.")
                continue

            values = struct.unpack('500H', data)
            values = np.array(values)
            values = values.reshape(batch_size, values_per_sample).astype(float)

            # print(values)
            # print(f1.shape, f2.shape, ax.shape, ay.shape, az.shape, gx.shape, gy.shape, gz.shape)

            state = np.concatenate([state, values], axis=0)[-n_samples:]
            # print(state.shape)

            # print(state[-100], state[-101], state.shape)
            # print(state[:500])
            state_features = get_model_features(state)

            # Make predictions
            predictions = model.predict(state_features)
            print("Predictions:", predictions)

            if predictions[0] == 'scratch':
                sender.send_data(b'scratch')

            # print(state_features)
            #v1 = cleanup(v1)
            #v2 = cleanup(v2)

            # print(round(np.mean(v1), 6), round(np.mean(v2), 6))
            
        except KeyboardInterrupt:
            print("Exiting...:")

            print(state)
            np.save(fname, state)
            reciever.close()
            sender.close()
            break

if __name__ == "__main__":
    main()