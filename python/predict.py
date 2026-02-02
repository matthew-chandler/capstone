import tensorflow as tf
import numpy as np
import librosa
import os
import sys
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
CLASSES = ['Yes', 'No', 'Unknown']
MODEL_PATH = 'kws_model.keras'

# --- 2. GET AUDIO FILE FROM COMMAND LINE ---
if len(sys.argv) < 2:
    print("Usage: python predict.py <path_to_audio_file> [--save-spec]")
    # As a fallback, use the default 'yes.wav'
    AUDIO_FILE_PATH = os.path.join('audio', 'yes.wav')
    print(f"No file provided, using default: {AUDIO_FILE_PATH}")
else:
    AUDIO_FILE_PATH = sys.argv[1]

# Check if we should save the spectrogram image
SAVE_SPECTROGRAM = "--save-spec" in sys.argv

if not os.path.exists(AUDIO_FILE_PATH):
    print(f"Error: Audio file not found at '{AUDIO_FILE_PATH}'")
    sys.exit(1)

# --- 3. PREPROCESSING FUNCTION (Copied from train.py) ---
# This function must be identical to the one used in training.
def get_mfcc(audio, label):
    # audio = tf.cast(audio, tf.float32) / 32768.0
    audio = audio[:16000] # Ensure audio is 1 second
    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)

    stft = tf.signal.stft(audio, frame_length=640, frame_step=320, fft_length=1024)
    spectrogram = tf.abs(stft)

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40, num_spectrogram_bins=stft.shape[-1], sample_rate=16000,
        lower_edge_hertz=20.0, upper_edge_hertz=4000.0)

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfcc = log_mel_spectrogram[..., tf.newaxis]
    return mfcc, label

# --- 4. LOAD MODEL AND AUDIO ---
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Loading audio file: {AUDIO_FILE_PATH}")
# librosa.load is great because it can resample to 16kHz automatically
audio_np, sample_rate = librosa.load(AUDIO_FILE_PATH, sr=16000, mono=True)


# --- 5. PREPROCESS AND PREDICT ---
# The 'label' is just a placeholder; it's not used in prediction.
mfcc, _ = get_mfcc(tf.constant(audio_np), tf.constant(0))

# The model expects a batch of inputs, so we add a batch dimension.
# Shape changes from (height, width, 1) to (1, height, width, 1)
mfcc_batch = tf.expand_dims(mfcc, 0)

print("Running prediction...")
prediction = model.predict(mfcc_batch)

# --- 6. VISUALIZE (if requested) ---
if SAVE_SPECTROGRAM:
    plt.figure(figsize=(10, 4))
    # Squeeze the batch and channel dimensions for visualization
    img_data = tf.squeeze(mfcc)
    # Transpose to have time on the x-axis, which is more conventional
    plt.imshow(tf.transpose(img_data), aspect='auto', origin='lower')
    plt.title(f"Log-Mel-Spectrogram for {os.path.basename(AUDIO_FILE_PATH)}")
    plt.ylabel("Mel Bins")
    plt.xlabel("Time Frames")
    plt.colorbar(format='%+2.0f dB')
    
    # Save the figure
    output_filename = f"spectrogram_{os.path.basename(AUDIO_FILE_PATH)}.png"
    plt.savefig(output_filename)
    print(f"✅ Spectrogram image saved to '{output_filename}'")


# --- 7. DISPLAY THE RESULT ---
pred_index = np.argmax(prediction[0])
pred_label = CLASSES[pred_index]
confidence = prediction[0][pred_index]

print(f"\n--> Prediction: '{pred_label}'")
print(f"    Confidence: {confidence*100:.2f}%")
