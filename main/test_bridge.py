from bridge import Bridge

reciever = Bridge("0.0.0.0", 12345, recieve=True)
reciever.connect()

sender = Bridge("172.20.10.22", 12345)
sender.connect()

while True:
    try:
        # i = input("Enter data to send: ")
        data = reciever.receive_data()
        print("Sending response...")
        if data[0:3] == "EMG":
            sender.send_data("True")
        else:
            reciever.connect()
            time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting...")
        reciever.close()
        sender.close()
        break