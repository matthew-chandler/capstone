import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# --- 1. CONFIGURATION ---
BATCH_SIZE = 32
# Must match the training script exactly
CLASSES = ['Yes', 'No', 'Unknown'] 

# --- 2. RE-DEFINE PREPROCESSING ---
# We need this function to convert raw audio into the shape the model expects.
def get_mfcc(audio, label):
    audio = tf.cast(audio, tf.float32) / 32768.0
    audio = audio[:16000]
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

def map_labels(audio, label):
    # Same mapping: 9->0 (Yes), 3->1 (No), Else->2 (Unknown)
    is_yes = tf.equal(label, 9)
    is_no = tf.equal(label, 3)
    new_label = tf.where(is_yes, 0, tf.where(is_no, 1, 2))
    return audio, new_label

# --- 3. LOAD DATA & MODEL ---
print("Loading Test Data...")
dataset = tfds.load('speech_commands', split='test', as_supervised=True)

# Apply the exact same pipeline
test_ds = dataset.map(map_labels)
test_ds = test_ds.map(get_mfcc)
test_ds = test_ds.batch(BATCH_SIZE)

print("Loading Saved Model...")
model = tf.keras.models.load_model('kws_model.keras')

# --- 4. EVALUATE ---
print("\nRunning Evaluation on Test Set...")
loss, acc = model.evaluate(test_ds)

print(f"\nFINAL TEST RESULTS:")
print(f"Accuracy: {acc*100:.2f}%")

# --- 5. VISUAL PREDICTION CHECK ---
print("\n--- SPOT CHECK (5 Random Examples) ---")
# Get a fresh batch
for mfccs, labels in test_ds.shuffle(100).take(1):
    predictions = model.predict(mfccs)
    
    # Show first 5 in the batch
    for i in range(5):
        true_label = CLASSES[labels[i]]
        
        # Get the highest confidence score
        pred_index = np.argmax(predictions[i])
        pred_label = CLASSES[pred_index]
        confidence = predictions[i][pred_index]
        
        # Status icon
        status = "✅" if true_label == pred_label else "❌"
        
        print(f"{status} True: {true_label:<10} | Pred: {pred_label:<10} ({confidence*100:.1f}%)")