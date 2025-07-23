from bridge import Bridge
import time
import numpy as np
import socket, struct

reciever = Bridge("0.0.0.0", 12345, recieve=True)
reciever.connect()

sender = Bridge("172.20.10.45", 12345)
sender.connect()

sample_rate = 1000
s_count = 0

while True:
    try:
        # i = input("Enter data to send: ")
        data = reciever.receive_data()
        values = struct.unpack('200H', data)

        v1 = np.array(values[0::2])
        v2 = np.array(values[1::2])

        v1 = v1 * 3.3 / 65535
        v2 = v2 * 3.3 / 65535

        print(len(v1), len(v2))
        s_count += 1

    except KeyboardInterrupt:
        print("Exiting... with count:", s_count)
        reciever.close()
        sender.close()
        break