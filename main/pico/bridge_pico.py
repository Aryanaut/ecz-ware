import socket
import time

class Bridge:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):

        """Establish a connection to the bridge."""

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.socket.connect((self.host, self.port))
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
        data = self.socket.recvfrom(1024).decode()
        print(f"Received data: {data}")
        return data

    def close(self):

        """Close the connection to the bridge."""
        
        print("Connection closed.")