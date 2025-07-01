import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from glob import glob

st.set_page_config(layout="wide")
st.title("📡 RF Signal Modulation Classifier")

RAW_DATA_DIR = "data/raw"
MODEL_PATH = "models/rf_modulation_classifier.pkl"
LABELS = ["BPSK", "QPSK", "16QAM"]

# Load model with error handling
@st.cache_resource
def load_classifier():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"❌ Model file not found at `{MODEL_PATH}`.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_classifier()

# List .npy files
available_files = sorted(glob(os.path.join(RAW_DATA_DIR, "*.npy")))
file_names = [os.path.basename(f) for f in available_files]

# File selector with error check
st.sidebar.header("Upload or Select File")
if not file_names:
    st.error("No `.npy` files found in `data/raw/`. Please upload at least one I/Q file.")
    st.stop()

selected_file = st.sidebar.selectbox("Choose an I/Q .npy file:", file_names)

# Load I/Q file
@st.cache_data
def load_iq_file(path):
    arr = np.load(path)
    if arr.shape[0] != 2:
        raise ValueError("Expected shape (2, N) for I/Q data")
    return arr.T  # shape: (N, 2)

try:
    iq_array = load_iq_file(os.path.join(RAW_DATA_DIR, selected_file))
except Exception as e:
    st.error(f"❌ Failed to load I/Q file: {e}")
    st.stop()

# Segment signal into overlapping chunks
segment_length = 128
segments = [
    iq_array[i:i+segment_length]
    for i in range(0, len(iq_array) - segment_length + 1, segment_length)
]
segments = np.array(segments).reshape(len(segments), -1)  # flatten

# Predict with model
try:
    predicted_indices = model.predict(segments)
    predicted_labels = [LABELS[i] for i in predicted_indices]
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

st.success("✅ Predictions complete.")
most_common = max(set(predicted_labels), key=predicted_labels.count)

# Signal summary
st.markdown(f"""
### 📊 Signal Summary:
- File: `{selected_file}`
- Samples: `{len(iq_array)}`
- Segments: `{len(segments)}`
- Most common modulation: **{most_common}**
""")

# Prediction Table
st.subheader("🧾 Segment Predictions")
st.dataframe({
    "Segment": list(range(len(predicted_labels))),
    "Predicted Modulation": predicted_labels
})

# Time Series Plot
st.subheader("📈 I/Q Time Series")
fig1, axs = plt.subplots(2, 1, figsize=(12, 3), sharex=True)
axs[0].plot(iq_array[:, 0], label="I")
axs[0].set_title("In-phase (I)")
axs[1].plot(iq_array[:, 1], label="Q", color="orange")
axs[1].set_title("Quadrature (Q)")
plt.tight_layout()
st.pyplot(fig1)

# Constellation Plot
st.subheader("✨ I/Q Constellation Plot")
fig2 = plt.figure(figsize=(5, 5))
plt.scatter(iq_array[:, 0], iq_array[:, 1], alpha=0.3, s=1)
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("I/Q Constellation")
st.pyplot(fig2)

# Spoofing detection
if len(set(predicted_indices)) > 1:
    st.warning("⚠️ Detected possible signal spoofing or modulation hopping.")
else:
    st.info("✅ No abnormal modulation behavior detected.")
