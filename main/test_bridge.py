from bridge import Bridge
import time
import numpy as np
import socket, struct
from sampler import *
import numpy as np

reciever = Bridge("0.0.0.0", 12345, recieve=True)
reciever.connect()

sender = Bridge("172.20.10.45", 12345)
sender.connect()

sample_rate = 1000
s_count = 0

fname = "ariel_rest.npy"

batch_size = 100
values_per_sample = 5
expected_bytes = batch_size * values_per_sample * 2  # 2 bytes per unsigned short

valuess = np.empty((0, values_per_sample))

while True:
    try:
        if len(valuess) == 300000:
            print("Exiting... with count:", s_count)

            print(valuess.shape)
            np.save(fname, valuess)
            reciever.close()
            sender.close()
            break
        # print("I'm trying man")
        # i = input("Enter data to send: ")
        data = reciever.receive_data()

        if len(data) != expected_bytes:
            print(f"Warning: received {len(data)} bytes, expected {expected_bytes}. Skipping.")
            continue

        values = struct.unpack('500H', data)
        values = np.array(values)
        values = values.reshape(batch_size, values_per_sample)

        if len(valuess) % 5000 == 0:
            print(f"Received {len(valuess)} samples, shape: {valuess.shape}")

        # print(values)
        # print(f1.shape, f2.shape, ax.shape, ay.shape, az.shape, gx.shape, gy.shape, gz.shape)

        value_stack = values

        if valuess.size == 0:
            valuess = value_stack
        else:
            valuess = np.vstack((valuess, value_stack))
        

        #v1 = cleanup(v1)
        #v2 = cleanup(v2)

        # print(round(np.mean(v1), 6), round(np.mean(v2), 6))
        
    except KeyboardInterrupt:
        print("Exiting...:")

        print(valuess.shape)
        np.save(fname, valuess)
        reciever.close()
        sender.close()
        break