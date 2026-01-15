import librosa
import numpy as np

# 1. Load the audio file (e.g., from Google Speech Commands)
# Sampling rate (sr) for Speech Commands is typically 16000 Hz
audio_path = 'path_to_your_audio_sample.wav'
y, sr = librosa.load(audio_path, sr=16000)

# 2. Define parameters based on your proposal
# 32ms window (approx. 512 samples at 16kHz) with a 50% overlap 
n_fft = 512
hop_length = 256  # 50% overlap
n_mels = 40       # Number of Mel bands
n_mfcc = 13       # Number of coefficients to keep

# 3. Generate MFCCs [cite: 16, 30]
# This function internally handles the FFT, Mel Filterbanks, and DCT [cite: 31]
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, 
                             n_fft=n_fft, hop_length=hop_length, 
                             n_mels=n_mels)

# 4. Display the results for the first audio packet
print(f"MFCC Shape: {mfccs.shape}") # (n_mfcc, number_of_frames)
print(f"First Frame Coefficients:\n{mfccs[:, 0]}")