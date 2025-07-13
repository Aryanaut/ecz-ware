# live_classify.py
import serial, time, joblib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# config
port, baudrate = 'COM3', 115200
window_size = 100        # smaller window ≈ faster response
max_plot = 300

model_file = input("model filename in models/ (e.g., rf_1720.joblib): ")
model = joblib.load(f"models/{model_file}")

ser = serial.Serial(port, baudrate, timeout=1)
buffer1 = deque([0]*window_size, maxlen=window_size)
buffer2 = deque([0]*window_size, maxlen=window_size)
plot1 = deque([0]*max_plot, maxlen=max_plot)
plot2 = deque([0]*max_plot, maxlen=max_plot)
pred_label = [0]

# matplotlib setup
fig, ax = plt.subplots()
line1, = ax.plot(range(max_plot), plot1, label='v1')
line2, = ax.plot(range(max_plot), plot2, label='v2')
text = ax.text(0.5,0.5,'', transform=ax.transAxes, ha='center', va='center', color='red', fontsize=24)
ax.set_ylim(0, 3.5)
ax.set_title('live emg')
ax.legend()

def update(frame):
    try:
        while ser.in_waiting:
            raw = ser.readline().decode('utf-8').strip()
            v1_str, v2_str = raw.split(',')
            v1, v2 = float(v1_str), float(v2_str)
            buffer1.append(v1)
            buffer2.append(v2)
            plot1.append(v1)
            plot2.append(v2)

        # only predict if enough data
        if len(buffer1) == window_size:
            feat = []
            for w in [buffer1, buffer2]:
                w = np.array(w)
                feat += [
                    np.mean(w), np.std(w), np.var(w),
                    np.sqrt(np.mean(np.square(w))),
                    np.sum(np.abs(np.diff(w)))
                ]
            pred = model.predict([feat])[0]
            pred_label[0] = pred

        # update plot + text
        text.set_text('SCRATCHING' if pred_label[0]==1 else '')
        line1.set_ydata(plot1)
        line2.set_ydata(plot2)

    except Exception as e:
        print("error:", e)

    return line1, line2, text

ani = animation.FuncAnimation(fig, update, interval=30)  # faster updates
plt.show()
ser.close()
