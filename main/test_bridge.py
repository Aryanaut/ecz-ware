from bridge import Bridge
import time
import numpy as np

reciever = Bridge("0.0.0.0", 12345, recieve=True)
reciever.connect()

sender = Bridge("172.20.10.22", 12345)
sender.connect()

sample_rate = 1000

while True:
    try:
        # i = input("Enter data to send: ")
        data = reciever.receive_data()
        print("Sending response...")
        if data[0:3] == "EMG":
            sender.send_data("True")
        
            sample_count = 0
            data_split = data.split("#")
            v1 = float(data_split[0][3:])
            v2 = float(data_split[1])

            print(v1, v2)

            while sample_count < sample_rate:
                sample_count += 1

        else:
            reciever.connect()
            time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting...")
        reciever.close()
        sender.close()
        break