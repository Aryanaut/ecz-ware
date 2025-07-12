import socket
import time

class Bridge:
    def __init__(self, host, port, recieve=False):
        self.host = host
        self.port = port
        self.socket = None
        self.recieve = recieve # couldn't think of a better way of making this bridge two way

    def connect(self):

        """Establish a connection to the bridge."""

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.socket.connect((self.host, self.port))
        if self.recieve:
            self.socket.bind(("0.0.0.0", self.port)) # Open on all channels I think
        print(f"Connected to bridge at {self.host}:{self.port}")

    def send_data(self, data):

        """Send data to the bridge."""

        if not self.socket:
            raise ConnectionError("Socket is not connected.")
            
        self.socket.sendto(data.encode(), (self.host, self.port))
        print(f"Sent data: {data}")

    def receive_data(self):

        """Receive data from the bridge."""

        if not self.socket:
            raise ConnectionError("Socket is not connected.")
        data, addr = self.socket.recvfrom(1024)
        print(f"Received data: {data.decode()}")
        return data.decode()

    def close(self):

        """Close the connection to the bridge."""
        
        print("Connection closed.")