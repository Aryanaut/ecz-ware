# record.py
import tkinter as tk
import csv, time, os, threading
import serial
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EMGRecorder:
    def __init__(self, port='COM3', baudrate=115200, maxlen=300):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.is_recording = False
        self.current_label = 0
        self.data = []
        self.buffer1 = deque([0]*maxlen, maxlen=maxlen)
        self.buffer2 = deque([0]*maxlen, maxlen=maxlen)

    def start_recording(self):
        os.makedirs('data', exist_ok=True)
        self.is_recording = True
        threading.Thread(target=self.record_loop, daemon=True).start()

    def stop_recording(self):
        self.is_recording = False
        filename = f"data/emg_{int(time.time())}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','v1','v2','label'])
            writer.writerows(self.data)
        print(f"saved to {filename}")

    def record_loop(self):
        while self.is_recording:
            try:
                raw = self.ser.readline().decode('utf-8').strip()
                v1_str, v2_str = raw.split(',')
                v1, v2 = float(v1_str), float(v2_str)
                timestamp = time.time()
                self.data.append([timestamp, v1, v2, self.current_label])
                self.buffer1.append(v1)
                self.buffer2.append(v2)
            except: continue

def run_ui():
    rec = EMGRecorder()

    def start(): rec.start_recording()
    def stop(): rec.stop_recording()
    def set_label0(): rec.current_label=0; label_var.set('label: not scratching')
    def set_label1(): rec.current_label=1; label_var.set('label: scratching')

    root = tk.Tk()
    root.title("emg recorder")

    label_var = tk.StringVar(value='label: not scratching')

    tk.Button(root, text='start recording', command=start).pack()
    tk.Button(root, text='stop recording', command=stop).pack()
    tk.Label(root, textvariable=label_var).pack()
    tk.Button(root, text='set label: not scratching', command=set_label0).pack()
    tk.Button(root, text='set label: scratching', command=set_label1).pack()

    # matplotlib plot
    fig, ax = plt.subplots(figsize=(5,3))
    line1, = ax.plot(range(len(rec.buffer1)), rec.buffer1, label='v1')
    line2, = ax.plot(range(len(rec.buffer2)), rec.buffer2, label='v2')
    ax.set_ylim(0, 3.5)
    ax.set_title('live emg')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    def update_plot():
        line1.set_ydata(rec.buffer1)
        line2.set_ydata(rec.buffer2)
        canvas.draw()
        root.after(50, update_plot)

    update_plot()
    root.mainloop()

if __name__ == "__main__":
    run_ui()
