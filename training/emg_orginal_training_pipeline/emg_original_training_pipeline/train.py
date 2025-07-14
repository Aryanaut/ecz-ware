# train.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib, os, time

def train(npzfile):
    data = np.load(f"features/{npzfile}")
    X, y = data['X'], data['y']
    print("X shape:", X.shape)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    print(classification_report(y, model.predict(X)))

    os.makedirs('models', exist_ok=True)
    out = f"models/rf_{int(time.time())}.joblib"
    joblib.dump(model, out)
    print(f"saved to {out}")

if __name__ == "__main__":
    file = input("npz filename (e.g., feat_1720000000.npz): ")
    train(file)
