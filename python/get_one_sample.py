import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

print("Loading model and dataset...")
model = tf.keras.models.load_model('kws_model.h5')
dataset = tfds.load('speech_commands', split='test', as_supervised=True)

# 1. Grab the first "Yes" audio clip (Label 9 in original dataset)
for audio, label in dataset:
    if label == 9: 
        raw_audio = audio
        break

# 2. Format to exactly 16000 floats
audio_float = tf.cast(raw_audio, tf.float32) / 32768.0
audio_float = audio_float[:16000]
padding = tf.zeros([16000] - tf.shape(audio_float), dtype=tf.float32)
audio_float = tf.concat([audio_float, padding], 0)

# 3. Run the exact DSP math to get the Python prediction
stft = tf.signal.stft(audio_float, frame_length=640, frame_step=320, fft_length=1024)
spectrogram = tf.abs(stft)
mel_matrix = tf.signal.linear_to_mel_weight_matrix(40, 513, 16000, 20.0, 4000.0)
mel_spectrogram = tf.tensordot(spectrogram, mel_matrix, 1)
log_mel = tf.math.log(mel_spectrogram + 1e-6)

# Reshape for the CNN: (1 batch, 49 frames, 40 bins, 1 channel)
mfcc_input = tf.expand_dims(log_mel[..., tf.newaxis], 0)
preds = model.predict(mfcc_input, verbose=0)[0]

# --- WRITE THESE NUMBERS DOWN ---
print("\n--- PYTHON GROUND TRUTH ---")
print(f"Yes:     {preds[0]:.6f}")
print(f"No:      {preds[1]:.6f}")
print(f"Unknown: {preds[2]:.6f}")
print("---------------------------\n")

# 4. Export the raw audio to a C header
print("Exporting audio to test_audio.h...")
with open("test_audio.h", "w") as f:
    f.write("/* AUTO-GENERATED TEST AUDIO ('YES') */\n")
    f.write("#ifndef TEST_AUDIO_H\n#define TEST_AUDIO_H\n\n")
    f.write("const float TEST_AUDIO[16000] = {\n")
    
    audio_np = audio_float.numpy()
    for i in range(0, len(audio_np), 10):
        vals = [f"{x:.6f}f" for x in audio_np[i:i+10]]
        f.write("    " + ", ".join(vals) + ",\n")
        
    f.write("};\n\n#endif // TEST_AUDIO_H\n")
print("✅ Done!")