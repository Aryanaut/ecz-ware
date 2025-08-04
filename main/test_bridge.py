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

fname = "aryan_extensor_rest6_0408.npy"

valuess = np.array([])

while True:
    try:
        if len(valuess) == 60000:
            print("Exiting... with count:", s_count)

            print(valuess.shape)
            np.save(fname, valuess)
            reciever.close()
            sender.close()
            break
        # print("I'm trying man")
        # i = input("Enter data to send: ")
        data = reciever.receive_data()
        values = struct.unpack('200H', data)
        values = np.array(values)
        values = (values / 65535) * 3.3

        v1 = np.array(values[0::2])
        v2 = np.array(values[1::2])

        v1 = np.round(v1, 6)
        v2 = np.round(v2, 6)

        if len(valuess) % 5000 == 0:
            print("recorded samples: ", len(valuess))
        # print(v)

        value_stack = np.column_stack([v1, v2])

        if valuess.size == 0:
            valuess = value_stack
        else:
            valuess = np.vstack((valuess, value_stack))
        

        #v1 = cleanup(v1)
        #v2 = cleanup(v2)

        # print(round(np.mean(v1), 6), round(np.mean(v2), 6))
        
        s_count += 1

    except KeyboardInterrupt:
        print("Exiting... with count:", s_count)

        print(valuess.shape)
        np.save(fname, valuess)
        reciever.close()
        sender.close()
        break