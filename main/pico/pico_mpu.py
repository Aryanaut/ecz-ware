from machine import ADC, I2C, Pin

# I2C (MPU6050)
i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=400000)
# ---------------- Tunables ----------------
ALPHA = 0.3
sample_interval = 0.02
required_consecutive = 3
THRESH_FRACTION = 0.45
HYSTERESIS_FRAC = 0.08
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
GYRO_XOUT_H = 0x43
MPU_ALPHA = 0.3
MPU_OSCILLATION_RESET_MS = 2000
MPU_REQUIRED_OSC = 5
MPU_MIN_COOLDOWN_MS = 1500
ACCEL_XOUT_H = 0x3B

# ---------------- Helper: MPU functions ----------------
def wake_mpu():
    try:
        i2c.writeto_mem(MPU6050_ADDR, PWR_MGMT_1, b'\x00')
    except:
        pass
    
def bytes_to_int(high, low):
        val = (high << 8) | low
        if val >= 0x8000:
            val = -((65535 - val) + 1)
        return val
    
def read_mpu6050_accel():
    raw_data = i2c.readfrom_mem(MPU6050_ADDR, ACCEL_XOUT_H, 6)  # only 6 bytes (ax, ay, az)

    ax = bytes_to_int(raw_data[0], raw_data[1])
    ay = bytes_to_int(raw_data[2], raw_data[3])
    az = bytes_to_int(raw_data[4], raw_data[5])
    
    return [ax, ay, az]

def low_pass_filter(prev, current, alpha):
    return alpha * current + (1 - alpha) * prev

def get_filtered_accel():
    global filtered_ax, filtered_ay, filtered_az
    ax, ay, az = read_mpu6050_accel()

    filtered_ax = low_pass_filter(filtered_ax, ax, ALPHA)
    filtered_ay = low_pass_filter(filtered_ay, ay, ALPHA)
    filtered_az = low_pass_filter(filtered_az, az, ALPHA)

    return int(filtered_ax), int(filtered_ay), int(filtered_az)

last_accel = [0, 0, 0]
def safe_read_accel():
    global last_accel
    try:
        accel = read_mpu6050_accel()
        last_accel = accel  # update buffer on success
        return accel
    except OSError as e:
        # On failure, log and return last known reading
        print("MPU6050 read failed:", e, "- using last known value")
        utime.sleep_ms(50)
        wake_mpu()  # recreate I2C and wake up MPU
        return last_accel
    
def full_mpu_reset():
    print("🔄 Full MPU6050 reset")
    try:
        i2c.writeto_mem(MPU6050_ADDR, PWR_MGMT_1, b'\x80')  # reset bit
        time.sleep_ms(150)
        i2c.writeto_mem(MPU6050_ADDR, PWR_MGMT_1, b'\x00')  # wake up
        time.sleep_ms(150)
    except Exception as e:
        print("Error during full MPU reset:", e)

def recover_i2c_bus():
    global i2c
    print("🔄 Recovering I2C bus...")
    try:
        i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=400000)
        time.sleep_ms(150)
    except Exception as e:
        print("Failed to recover I2C bus:", e)