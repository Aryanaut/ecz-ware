import network
import usocket as socket
import secrets
from bridge_pico import Bridge
import time

# connect to my phone's hotspot
# change the SSID and password in secrets.py if you want to change to your own hotspot
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
SSID, PWD = secrets.networks[2]
wlan.connect(SSID, PWD)
print("Connecting to...", SSID, PWD)

wlan.ifconfig((secrets.SERVER, secrets.SUBNET, secrets.GATEWAY, secrets.DNS))

while not wlan.isconnected():
    pass

print("Connection Successful")
print(wlan.ifconfig()[0])

b = Bridge(secrets.CLIENT, secrets.PORT)
b.connect()

while True:
    b.send_data("Hello")
    time.sleep(2)
    

