from machine import Pin
import time

led = Pin(25)

def connected():
    led.on()
    time.sleep(0.5)
    led.off()
    time.sleep(0.5)
    led.on()
    time.sleep(0.5)
    led.off()

def disconnected():
    led.on()
    time.sleep(2)
    led.off()

def sent_data():
    led.on()
    time.sleep(0.1)
    led.off()

def received_data():
    led.on()
    time.sleep(0.5)
    led.off()