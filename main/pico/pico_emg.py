from machine import ADC, Pin
from bridge_pico import Bridge
import utime
import pico_secrets
from pico_animations import *

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
SSID, PWD = secrets.networks[2]
wlan.connect(SSID, PWD)
print("Connecting to...", SSID, PWD)


adc1 = ADC(Pin(26))  # EMG sensor 1
adc2 = ADC(Pin(27))  # EMG sensor 2

sender = Bridge("172.20.10.2", 12345)
sender.connect()
print("Sender connected.")

reciever = Bridge(wlan.ifconfig()[0], 12345, recieve=True)
reciever.connect()
print("Receiver connected.")

while True:
    v1 = adc1.read_u16() * 3.3 / 65535
    v2 = adc2.read_u16() * 3.3 * 2 / 65535

    sender.send_data(v1, v2)
    reciever.receive_data()

    if reciever.receive_data() == "True":
        print("Data received on Pi5.")
        pico_animations.received_data()

    print(f"{v1:.3f},{v2:.3f}")
    utime.sleep_ms(10)