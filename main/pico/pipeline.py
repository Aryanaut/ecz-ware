from machine import ADC, I2C, Pin
from bridge_pico import Bridge
from pico_mpu import *
import utime
import pico_secrets
import pico_animations
import network, usocket as socket, time, struct

adc_idx = ADC(26)    # GP26 ADC0
adc_ring = ADC(27)   # GP27 ADC1

# Haptic outputs
h1 = Pin(3, Pin.OUT)
h2 = Pin(4, Pin.OUT)
h3 = Pin(5, Pin.OUT)

# Pico W onboard LED
led = Pin("LED", Pin.OUT)

# ---------------- Manual flex thresholds ----------------
cal = {
    'idx_flat': 31608,
    'idx_bent': 25401,
    'ring_flat': 27474,
    'ring_bent': 21383
}

# Haptic activation
HAPTIC_ON_MS = 300
HAPTIC_COOLDOWN_MS = 1000

# ---------------- Haptic helper ----------------
def haptics_pulse(ms=HAPTIC_ON_MS):
    h1.on(); h2.on(); h3.on()
    led.value(1)
    time.sleep_ms(ms)
    h1.off(); h2.off(); h3.off()
    led.value(0)

# ---------------- Main loop ----------------
wake_mpu()
last_haptic_time = 0

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
SSID, PWD = pico_secrets.networks[0]
wlan.connect(SSID, PWD)
print("Connecting to...", SSID, PWD)

wlan.ifconfig((pico_secrets.SERVER, pico_secrets.SUBNET, pico_secrets.GATEWAY, pico_secrets.DNS))
while not wlan.isconnected():
    pass

sender = Bridge("172.20.10.2", 12345)
sender.connect()
print("Sender connected.")

reciever = Bridge(wlan.ifconfig()[0], 12345, recieve=True)
reciever.connect()
print("Receiver connected.")

pico_animations.connected()

interval = 1 / 1000
start_time = time.ticks_ms()

batch_size = 100

while True:
    batch = []
    for i in range(batch_size):
        v1 = adc_idx.read_u16()
        v2 = adc_ring.read_u16()
        # print(str(v1 * 3.3 / (65535)), str(v2 * 3.3 / (65535)))
        filt_axes = safe_read_accel()
        batch.extend([v1, v2])
        batch.extend(filt_axes)
        time.sleep(interval)

    packet = struct.pack(f'{len(batch)}H', *batch)
    try:
        sender.send_data(packet)
    except Exception as e:
        print(e)
