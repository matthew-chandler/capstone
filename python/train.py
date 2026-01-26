import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SETUP & CONSTANTS ---
BATCH_SIZE = 32
EPOCHS = 10
# We map labels to: 0=Yes, 1=No, 2=Unknown
CLASSES = ['Yes', 'No', 'Unknown']

# Load Dataset
print("Loading Speech Commands...")
dataset, info = tfds.load('speech_commands', with_info=True, as_supervised=True)
train_raw = dataset['train']
val_raw = dataset['validation']

# --- 2. PREPROCESSING FUNCTIONS ---

def map_labels(audio, label):
    """
    Maps the original 12 classes to our 3 target classes:
    Original 9 (Yes) -> 0
    Original 3 (No)  -> 1
    Everything else  -> 2
    """
    # TensorFlow logic (must use tf.equal, not ==)
    is_yes = tf.equal(label, 9)
    is_no = tf.equal(label, 3)
    
    # Nested if-else: If Yes(0), else if No(1), else Unknown(2)
    new_label = tf.where(is_yes, 0, tf.where(is_no, 1, 2))
    return audio, new_label

def get_mfcc(audio, label):
    # 1. Normalize and Pad
    audio = tf.cast(audio, tf.float32) / 32768.0
    audio = audio[:16000] # Trim if long
    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)

    # 2. STFT -> Spectrogram
    stft = tf.signal.stft(audio, frame_length=640, frame_step=320, fft_length=1024)
    spectrogram = tf.abs(stft)

    # 3. Mel Spectrogram
    # Create the filter bank (matrix)
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40, num_spectrogram_bins=stft.shape[-1], sample_rate=16000,
        lower_edge_hertz=20.0, upper_edge_hertz=4000.0)
    
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    
    # 4. Log Scale (This is what makes it look like an image)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    
    # 5. Add Channel Dimension (Required for CNN input)
    # Shape becomes: (Time, Freq, 1)
    mfcc = log_mel_spectrogram[..., tf.newaxis]
    
    return mfcc, label

# --- 3. BUILD DATA PIPELINE ---
print("Building data pipeline...")
# Train Data
train_ds = train_raw.map(map_labels)
train_ds = train_ds.map(get_mfcc)
train_ds = train_ds.cache() # Keep in memory for speed
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Validation Data
val_ds = val_raw.map(map_labels)
val_ds = val_ds.map(get_mfcc)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Get input shape from one example
for example_audio, example_label in train_ds.take(1):
    input_shape = example_audio.shape[1:]
    print(f"Model Input Shape: {input_shape}")

# --- 4. DEFINE THE CNN MODEL ---
model = tf.keras.models.Sequential([
    # Input Layer
    tf.keras.layers.Input(shape=input_shape),
    
    # First Convolution Block
    # Resizing ensures input is uniform, though our padding step handles most of it
    tf.keras.layers.Resizing(32, 32), 
    tf.keras.layers.Normalization(),
    
    # Conv2D: Scans the MFCC image for features
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    # Second Convolution Block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    # Flatten & Dense
    tf.keras.layers.Dropout(0.25), # Helps prevent overfitting
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    # Output Layer (3 Neurons = Yes, No, Unknown)
    tf.keras.layers.Dense(3, activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 5. TRAIN ---
print("\nStarting Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)]
)

# --- 6. SAVE ---
model.save("kws_model.keras")
print("✅ Model saved to 'kws_model.keras'")