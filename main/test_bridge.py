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

valuess = np.array([])

while True:
    try:
        # print("I'm trying man")
        # i = input("Enter data to send: ")
        data = reciever.receive_data()
        values = struct.unpack('1H', data)

        v = (values[0] / 65535) * 3.3
        print(v)

        valuess = np.append(valuess, v)

        #v1 = cleanup(v1)
        #v2 = cleanup(v2)

        # print(round(np.mean(v1), 6), round(np.mean(v2), 6))
        
        s_count += 1

    except KeyboardInterrupt:
        print("Exiting... with count:", s_count)

        print(valuess.shape)
        np.save('other.npy', valuess)
        reciever.close()
        sender.close()
        break