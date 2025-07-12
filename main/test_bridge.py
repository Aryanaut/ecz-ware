from bridge import Bridge

b = Bridge("172.20.10.22", 12345, recieve=True)
b.connect()

while True:
    try:
        # i = input("Enter data to send: ")
        b.receive_data()
    except KeyboardInterrupt:
        print("Exiting...")
        b.close()
        break