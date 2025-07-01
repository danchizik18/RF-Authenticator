import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from glob import glob

st.set_page_config(layout="wide")
st.title("📡 RF Signal Modulation Classifier")

# Paths
RAW_DATA_DIR = "data/raw"
MODEL_PATH = "models/rf_modulation_classifier.joblib"  # Make sure this exists
LABELS = ["BPSK", "QPSK", "16QAM"]

# Load model
@st.cache_resource
def load_classifier():
    return joblib.load(MODEL_PATH)

model = load_classifier()

# List available .npy files
available_files = sorted(glob(os.path.join(RAW_DATA_DIR, "*.npy")))
file_names = [os.path.basename(f) for f in available_files]

# Sidebar controls
st.sidebar.header("Upload or Select File")
selected_file = st.sidebar.selectbox("Choose an I/Q .npy file:", file_names)

# Load data
@st.cache_data
def load_iq_file(path):
    arr = np.load(path)
    if arr.shape[0] != 2:
        raise ValueError("Expected shape (2, N) for I/Q data")
    iq = arr.T  # shape: (N, 2)
    return iq

iq_array = load_iq_file(os.path.join(RAW_DATA_DIR, selected_file))

# Segment into overlapping windows of 128 samples
segment_length = 128
segments = [iq_array[i:i+segment_length] for i in range(0, len(iq_array) - segment_length + 1, segment_length)]
segments = np.array(segments)  # shape: (n_segments, 128, 2)

# Predict
probs = model.predict(segments)
predicted_indices = np.argmax(probs, axis=1)
predicted_labels = [LABELS[i] for i in predicted_indices]

st.success("✅ Predictions complete.")

# Summary box
most_common = max(set(predicted_labels), key=predicted_labels.count)
st.markdown(f"""
### 📊 Signal Summary:
- File: `{selected_file}`
- Samples: `{len(iq_array)}`
- Segments: `{len(segments)}`
- Most common modulation: **{most_common}**
""")

# Table of predictions
st.subheader("🧾 Segment Predictions")
st.dataframe({
    "Segment": list(range(len(predicted_labels))),
    "Predicted Modulation": predicted_labels
})

# I/Q Time Domain Plots
st.subheader("📈 I/Q Time Series")
fig1, axs = plt.subplots(2, 1, figsize=(12, 3), sharex=True)
axs[0].plot(iq_array[:, 0], label="I")
axs[0].set_title("In-phase (I) Component")
axs[1].plot(iq_array[:, 1], label="Q", color="orange")
axs[1].set_title("Quadrature (Q) Component")
plt.tight_layout()
st.pyplot(fig1)

# I-Q Constellation
st.subheader("✨ I/Q Constellation Plot")
fig2 = plt.figure(figsize=(5, 5))
plt.scatter(iq_array[:, 0], iq_array[:, 1], alpha=0.3, s=1)
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("I/Q Constellation")
st.pyplot(fig2)

# Confidence per segment
st.subheader("📉 Prediction Confidence by Segment")
fig3, ax3 = plt.subplots(figsize=(10, 3))
for i, label in enumerate(LABELS):
    ax3.plot([p[i] for p in probs], label=label)
ax3.set_title("Prediction Confidence per Segment")
ax3.set_xlabel("Segment")
ax3.set_ylabel("Confidence")
ax3.legend()
st.pyplot(fig3)

# Spoofing / Anomaly detection
if len(set(predicted_indices)) > 1:
    st.warning("⚠️ Detected possible signal spoofing or modulation hopping.")
else:
    st.info("No abnormal modulation behavior detected.")
