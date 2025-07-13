from machine import ADC, Pin
import utime

adc1 = ADC(Pin(26))  # EMG sensor 1
adc2 = ADC(Pin(27))  # EMG sensor 2

while True:
    v1 = adc1.read_u16() * 3.3 / 65535
    v2 = adc2.read_u16() * 3.3 * 2/ 65535
    print(f"{v1:.3f},{v2:.3f}")
    utime.sleep_ms(10)