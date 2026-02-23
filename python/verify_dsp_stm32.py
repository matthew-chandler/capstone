import tensorflow as tf
import numpy as np

# 1. Create a Dummy Signal (A linear ramp from -1.0 to 1.0)
# This is better than a sine wave because it contains all frequencies
signal = np.linspace(-1.0, 1.0, 640, dtype=np.float32)

# 2. Apply Hann Window
hann = tf.signal.hann_window(640, periodic=True)
windowed = signal * hann

# 3. Pad to 1024 (Required for 1024-point FFT)
padded = tf.concat([windowed, tf.zeros(1024 - 640)], axis=0)

# 4. Execute FFT
# tf.signal.rfft computes the 1D real FFT.
fft_result = tf.signal.rfft(padded)
magnitudes = tf.abs(fft_result)  # Shape: 513

# 5. Mel Matrix
mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=40, num_spectrogram_bins=513, sample_rate=16000,
    lower_edge_hertz=20.0, upper_edge_hertz=4000.0)

# 6. Matrix Multiply & Log
mel_energies = tf.tensordot(magnitudes, mel_matrix, 1)
log_mel = tf.math.log(mel_energies + 1e-6)

# 7. Print Ground Truth
print("--- PYTHON GROUND TRUTH ---")
print("First 5 Log-Mel Values:")
for i in range(len(log_mel)):
    print(f"[{i}]: {log_mel[i]:.6f}")