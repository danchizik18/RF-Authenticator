import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from sklearn.model_selection import train_test_split

X, y = [], []
modulations = ["bpsk", "qpsk", "qam16"]

for i, mod in enumerate(modulations):
    data = np.load(f"data/raw/{mod}.npy")  # shape: (2, 1024)
    # Transpose to (1024, 2)
    data = data.T
    # Reshape to (samples, 128, 2)
    data = data[:1024 - 1024 % 128].reshape(-1, 128, 2)
    X.append(data)
    y.append(np.full((data.shape[0],), i))

X = np.vstack(X)
y = np.hstack(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv1D(16, kernel_size=3, activation="relu", input_shape=(128, 2)),
    Conv1D(32, kernel_size=3, activation="relu"),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(len(modulations), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save("models/rf_modulation_classifier.h5")
