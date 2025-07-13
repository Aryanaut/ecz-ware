from bridge import Bridge

reciever = Bridge("172.20.10.22", 12345, recieve=True)
receiver.connect()

sender = Bridge("172.20.10.22", 12345)
sender.connect()

while True:
    try:
        # i = input("Enter data to send: ")
        data = b.receive_data()
        if len(data[0]) == 2:


    except KeyboardInterrupt:
        print("Exiting...")
        b.close()
        break