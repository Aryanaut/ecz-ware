import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import serial
import time
import os
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === CONFIG ===
DATA_DIR = 'data'
MODEL_DIR = 'saved_models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Dual EMG App", layout="wide")
st.title("Dual EMG Guided Recording & Classification App")

# === SESSION STATE INIT ===
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'recorded_data' not in st.session_state:
    st.session_state.recorded_data = []
if 'ser' not in st.session_state:
    st.session_state.ser = None

# === TABS ===
tabs = st.tabs(["Guided Record Data", "Extract Features", "Train Model", "Live Classification"])

# === TAB 1: Guided Record Data ===
with tabs[0]:
    st.header("Guided Balanced EMG Recording (2 sensors)")

    port = st.text_input("Serial Port (e.g., COM3)", value='COM3')
    baudrate = st.number_input("Baudrate", value=115200)
    total_minutes = st.slider("Total Recording Time (minutes)", 3, 25, 5)
    cycle_seconds = 10

    # define classes
    class_options = [
        (0, "Rest"),
        (1, "Scratching"),
        (2, "Other")
    ]
    selected_classes = st.multiselect("Classes to include:", ["Rest","Scratching","Other"], default=["Rest","Scratching"])
    class_map = {name: num for num, name in class_options if name in selected_classes}
    label_cycle = list(class_map.items())

    if st.button("Start Guided Recording") and not st.session_state.is_recording:
        try:
            st.session_state.ser = serial.Serial(port, baudrate, timeout=1)
            st.session_state.is_recording = True
            st.session_state.recorded_data = []

            st.success("Recording started. Follow the instructions!")
            plot_placeholder = st.empty()
            prompt_placeholder = st.empty()

            total_seconds = total_minutes * 60
            cycle_idx = 0
            buffer1 = deque(maxlen=300)
            buffer2 = deque(maxlen=300)

            start_time = time.time()
            next_switch = start_time + cycle_seconds
            current_label_name, current_label_num = label_cycle[cycle_idx]

            while st.session_state.is_recording:
                try:
                    raw = st.session_state.ser.readline().decode('utf-8').strip()
                    v1_str, v2_str = raw.split(',')
                    v1 = float(v1_str)
                    v2 = float(v2_str)
                    timestamp = time.time()
                    st.session_state.recorded_data.append([timestamp, v1, v2, current_label_num])
                    buffer1.append(v1)
                    buffer2.append(v2)
                except:
                    pass

                # plot both
                fig, ax = plt.subplots()
                ax.plot(buffer1, label='Sensor 1')
                ax.plot(buffer2, label='Sensor 2')
                ax.set_ylim(0,3.5)
                ax.set_title("Live EMG Data (2 sensors)")
                ax.legend()
                plot_placeholder.pyplot(fig)

                prompt_placeholder.markdown(f"## **Do: {current_label_name.upper()}**")

                now = time.time()
                if now >= next_switch:
                    cycle_idx = (cycle_idx + 1) % len(label_cycle)
                    current_label_name, current_label_num = label_cycle[cycle_idx]
                    next_switch += cycle_seconds

                if now - start_time >= total_seconds:
                    break

                time.sleep(0.01)

            st.session_state.is_recording = False
            if st.session_state.ser:
                st.session_state.ser.close()
                st.session_state.ser = None

            # save CSV
            filename = f"{DATA_DIR}/emg_guided_dual_{int(time.time())}.csv"
            df = pd.DataFrame(st.session_state.recorded_data, columns=['timestamp','v1','v2','label'])
            df.to_csv(filename, index=False)
            st.success(f"Recording complete! Saved {len(df)} samples to {filename}")
            st.line_chart(df[['v1','v2']])

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.is_recording = False
            if st.session_state.ser:
                st.session_state.ser.close()
                st.session_state.ser = None

# === TAB 2: Extract Features ===
with tabs[1]:
    st.header("Extract Features (dual EMG)")
    raw_files = [f for f in os.listdir(DATA_DIR) if f.startswith('emg_guided_dual')]
    if raw_files:
        raw_file = st.selectbox("Select raw CSV", raw_files)
        window_size = st.number_input("Window Size", value=100)
        step_size = st.number_input("Step Size", value=50)
        if st.button("Extract"):
            df = pd.read_csv(f"{DATA_DIR}/{raw_file}")
            v1s = df['v1'].values
            v2s = df['v2'].values
            labels = df['label'].values
            X, y = [], []
            for start in range(0, len(v1s)-window_size+1, step_size):
                w1 = v1s[start:start+window_size]
                w2 = v2s[start:start+window_size]
                feats = [
                    np.mean(np.abs(w1)), np.sqrt(np.mean(w1**2)), np.var(w1), np.sum(np.abs(np.diff(w1))),
                    np.mean(np.abs(w2)), np.sqrt(np.mean(w2**2)), np.var(w2), np.sum(np.abs(np.diff(w2)))
                ]
                win_labels = labels[start:start+window_size]
                label = max(set(win_labels), key=list(win_labels).count)
                X.append(feats)
                y.append(label)
            feature_cols=['mav1','rms1','var1','wl1','mav2','rms2','var2','wl2']
            features_df = pd.DataFrame(X, columns=feature_cols)
            features_df['label'] = y
            feat_file = raw_file.replace('dual','features_dual')
            features_df.to_csv(f"{DATA_DIR}/{feat_file}", index=False)
            st.success(f"Saved to {feat_file}")
            st.dataframe(features_df.head())
    else:
        st.info("No raw data files found.")

# === TAB 3: Train Model ===
with tabs[2]:
    st.header("Train Model")
    feat_files = [f for f in os.listdir(DATA_DIR) if f.startswith('emg_guided_features_dual')]
    if feat_files:
        feat_file = st.selectbox("Select feature CSV", feat_files)
        model_type = st.selectbox("Model", ['RandomForest','ANN'])
        test_size = st.slider("Test split", 0.1,0.5,0.2)
        if st.button("Train"):
            df = pd.read_csv(f"{DATA_DIR}/{feat_file}")
            feature_cols=['mav1','rms1','var1','wl1','mav2','rms2','var2','wl2']
            X = df[feature_cols].values
            y = df['label'].values
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=42)
            if model_type=='RandomForest':
                model = RandomForestClassifier(n_estimators=100,random_state=42)
            else:
                model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write("Accuracy:", acc)
            st.text(classification_report(y_test, y_pred))
            model_path = f"{MODEL_DIR}/emg_model_dual_{model_type}_{int(time.time())}.pkl"
            joblib.dump(model, model_path)
            st.success(f"Model saved to {model_path}")
    else:
        st.info("No feature files found.")

# === TAB 4: Live Classification ===
with tabs[3]:
    st.header("Live Classification")
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    if model_files:
        model_file = st.selectbox("Select Model", model_files)
        live_port = st.text_input("Serial Port (live)", value='COM3', key='live_port')
        live_baud = st.number_input("Baudrate", value=115200, key='live_baud')
        window_size = st.number_input("Window Size (live)", value=100, key='live_win')
        step_size = st.number_input("Step Size (live)", value=50, key='live_step')
        if st.button("Start Live"):
            model = joblib.load(f"{MODEL_DIR}/{model_file}")
            ser = serial.Serial(live_port, live_baud, timeout=1)
            data1 = deque([0]*500, maxlen=500)
            data2 = deque([0]*500, maxlen=500)
            window1 = deque(maxlen=window_size)
            window2 = deque(maxlen=window_size)
            placeholder = st.empty()
            pred_placeholder = st.empty()
            step_counter=0
            try:
                while True:
                    raw = ser.readline().decode('utf-8').strip()
                    v1_str, v2_str = raw.split(',')
                    v1 = float(v1_str)
                    v2 = float(v2_str)
                    data1.append(v1)
                    data2.append(v2)
                    window1.append(v1)
                    window2.append(v2)
                    step_counter+=1
                    fig, ax = plt.subplots()
                    ax.plot(data1, label='Sensor 1')
                    ax.plot(data2, label='Sensor 2')
                    ax.set_ylim(0,3.5)
                    ax.set_title("Live EMG Data")
                    ax.legend()
                    placeholder.pyplot(fig)
                    if len(window1)==window_size and step_counter>=step_size:
                        feats=[
                            np.mean(np.abs(window1)), np.sqrt(np.mean(np.square(window1))),
                            np.var(window1), np.sum(np.abs(np.diff(window1))),
                            np.mean(np.abs(window2)), np.sqrt(np.mean(np.square(window2))),
                            np.var(window2), np.sum(np.abs(np.diff(window2)))
                        ]
                        pred = model.predict([feats])[0]
                        pred_placeholder.markdown(f"**Prediction:** {pred}")
                        step_counter=0
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                ser.close()
    else:
        st.info("No trained models found.")
