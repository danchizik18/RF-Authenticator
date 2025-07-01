# 📡 RF Signal Modulation Classifier

A lightweight, Streamlit-based web app that classifies radio frequency (RF) signal modulations — BPSK, QPSK, and 16QAM — from I/Q sample data using a convolutional neural network (CNN).

---

## 🚀 Features

- Upload or select `.npy` I/Q files for classification
- Real-time predictions of modulation type
- Signal visualizations: time series and I/Q constellation plots
- Basic spoofing/anomaly detection via modulation variance
- Interactive Streamlit dashboard with clean UI

---

## 🧠 Model

The classifier is a 1D CNN trained on synthetic baseband I/Q data for:

- **BPSK**
- **QPSK**
- **16QAM**

Each `.npy` file contains raw I/Q signals shaped `(2, N)` which are segmented and processed for classification.

---

## 📁 Project Structure

rf-authenticator/
├── dashboard.py # Streamlit web app
├── train_classifier.py # Model training script
├── models/
│ └── rf_modulation_classifier.h5 # Trained TensorFlow model
├── data/
│ └── raw/
│ ├── bpsk.npy
│ ├── qpsk.npy
│ └── qam16.npy
├── requirements.txt
└── .gitignore


---

## 🔧 Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/danchizik18/rf-authenticator-.git
cd rf-authenticator-

🧠 Author
Dan Chizik
Stats & Data Science @ UC Berkeley
GitHub: @danchizik18

