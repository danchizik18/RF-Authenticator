import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

modulations = ["bpsk", "qpsk", "qam16"]
X, y = [], []

for i, mod in enumerate(modulations):
    path = f"data/raw/{mod}.npy"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    data = np.load(path)  # shape: (2, 1024)
    data = data.T  # shape: (1024, 2)
    data = data[:1024 - 1024 % 128].reshape(-1, 128, 2)
    X.append(data)
    y.append(np.full((data.shape[0],), i))

X = np.vstack(X)
y = np.hstack(y)

X_flat = X.reshape(X.shape[0], -1)
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/rf_modulation_classifier.pkl")
print("✅ Model trained and saved to models/rf_modulation_classifier.pkl")
