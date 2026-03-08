import tensorflow as tf
import librosa
import numpy as np
import os

# Force legacy Keras to avoid the TFLite/Keras 3 bug
os.environ["TF_USE_LEGACY_KERAS"] = "1"

print("Loading model...")
model = tf.keras.models.load_model('kws_model.h5')

# The files we want to process and the variable names for the C header
wav_files = ["audio/yes.wav", "audio/no.wav", "audio/unknown.wav"]
array_names = ["TEST_AUDIO_YES", "TEST_AUDIO_NO", "TEST_AUDIO_UNKNOWN"]

header_content = "/* AUTO-GENERATED TEST AUDIO FROM WAV FILES */\n"
header_content += "#ifndef TEST_AUDIO_H\n#define TEST_AUDIO_H\n\n"

for wav_file, array_name in zip(wav_files, array_names):
    print(f"\nProcessing {wav_file}...")
    
    # 1. Load the audio and force it to 16kHz. 
    # Librosa automatically normalizes the audio to floats between -1.0 and +1.0
    audio, sr = librosa.load(wav_file, sr=16000)
    
    # 2. Pad or truncate to exactly 16,000 samples (1 second)
    if len(audio) > 16000:
        audio = audio[:16000]
    else:
        padding = 16000 - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
        
    # Convert to a TensorFlow tensor so we can run the exact same DSP math
    audio_float = tf.convert_to_tensor(audio, dtype=tf.float32)
    
    # 3. Run the exact DSP math to get the Python prediction
    stft = tf.signal.stft(audio_float, frame_length=640, frame_step=320, fft_length=1024)
    spectrogram = tf.abs(stft)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(40, 513, 16000, 20.0, 4000.0)
    mel_spectrogram = tf.tensordot(spectrogram, mel_matrix, 1)
    log_mel = tf.math.log(mel_spectrogram + 1e-6)

    # Reshape for the CNN: (1 batch, 49 frames, 40 bins, 1 channel)
    mfcc_input = tf.expand_dims(log_mel[..., tf.newaxis], 0)
    preds = model.predict(mfcc_input, verbose=0)[0]

    # Print the Ground Truth so you can compare it with the STM32
    print(f"--- PYTHON GROUND TRUTH: {wav_file} ---")
    print(f"Yes:     {preds[0]:.6f}")
    print(f"No:      {preds[1]:.6f}")
    print(f"Unknown: {preds[2]:.6f}")
    print("----------------------------------------")
    
    # 4. Format into a C array and append to the header string
    header_content += f"const float {array_name}[16000] = {{\n"
    for i in range(0, len(audio), 10):
        vals = [f"{x:.6f}f" for x in audio[i:i+10]]
        header_content += "    " + ", ".join(vals) + ",\n"
    header_content += "};\n\n"

header_content += "#endif // TEST_AUDIO_H\n"

# 5. Save the giant string to a single .h file
with open("test_audio.h", "w") as f:
    f.write(header_content)
    
print("\n✅ Successfully generated test_audio.h with all 3 arrays!")