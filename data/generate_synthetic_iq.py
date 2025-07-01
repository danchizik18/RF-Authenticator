import numpy as np
import os

SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_bpsk(num_samples=1024):
    bits = np.random.randint(0, 2, num_samples)
    symbols = 2 * bits - 1  # Map 0→-1, 1→+1
    iq = symbols + 0j
    return np.vstack([np.real(iq), np.imag(iq)])

def generate_qpsk(num_samples=1024):
    bits = np.random.randint(0, 4, num_samples)
    mapping = {
        0: 1 + 1j,
        1: -1 + 1j,
        2: -1 - 1j,
        3: 1 - 1j
    }
    symbols = np.array([mapping[b] for b in bits])
    iq = symbols / np.sqrt(2)  # Normalize power
    return np.vstack([np.real(iq), np.imag(iq)])

def generate_16qam(num_samples=1024):
    real = np.random.choice([-3, -1, 1, 3], num_samples)
    imag = np.random.choice([-3, -1, 1, 3], num_samples)
    iq = real + 1j * imag
    iq /= np.sqrt((np.abs(iq) ** 2).mean())  # Normalize power
    return np.vstack([np.real(iq), np.imag(iq)])

# Generate and save
np.save(os.path.join(SAVE_DIR, "bpsk.npy"), generate_bpsk())
np.save(os.path.join(SAVE_DIR, "qpsk.npy"), generate_qpsk())
np.save(os.path.join(SAVE_DIR, "qam16.npy"), generate_16qam())

print("✅ Saved synthetic IQ samples to data/raw/")
