import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt

# 1. Download and Load the Dataset
# This will download approx 2.4GB to your ~/tensorflow_datasets/ folder
# 'with_info=True' gives us metadata (like label names)
# 'as_supervised=True' gives us tuples of (audio, label) instead of a dictionary
print("Attempting to download/load the dataset. This usually takes 2-5 minutes...")
dataset, info = tfds.load('speech_commands', with_info=True, as_supervised=True)

# 2. Access the Training Split
train_data = dataset['train']

# 3. Fetch One Single Entry
# We use .take(1) to grab just the first example
for audio, label in train_data.take(4):
    
    # 4. Extract Key Details
    # The label is an integer ID (e.g., 5), we want the name (e.g., "go")
    label_name = info.features['label'].int2str(label)
    
    # Audio comes in as a Tensor, convert to numpy for inspection
    audio_np = audio.numpy()
    
    # 5. Output Verification Stats
    print("\n" + "="*40)
    print("✅ DATASET VERIFICATION SUCCESSFUL")
    print("="*40)
    print(f"Label ID:   {label}")
    print(f"Command:    '{label_name}'")
    print(f"Shape:      {audio_np.shape}")
    print(f"Data Type:  {audio_np.dtype}")
    print(f"Min Value:  {np.min(audio_np)}")
    print(f"Max Value:  {np.max(audio_np)}")
    print("="*40 + "\n")    

    # Optional: Warn if audio is not 1 second (16000 samples)
    if audio_np.shape[0] != 16000:
        print(f"⚠️ Note: This clip is {audio_np.shape[0]} samples long (not standard 16000).")

    ### MFCC TRANSFORM ###

    # 1. convert audio sample to float bc that's what the library expects
    audio_float = audio_np.astype(np.float32) / 32768.0
    print(f"float audio shape: {audio_float.shape}")
    print(f"float audio sample values: {audio_float[:5]}")

    # 2. Define Parameters
    SAMPLE_RATE = 16000
    N_MFCC = 40       # How many rows (height of image)
    N_FFT = 1024      # Resolution of the frequency analysis
    HOP_LENGTH = 160  # How much we slide the window (160 samples = 10ms)

    # 3. Compute MFCC
    # This function does the Windowing -> FFT -> Mel Filter -> Log -> DCT
    mfcc = librosa.feature.mfcc(
        y=audio_float, 
        sr=SAMPLE_RATE, 
        n_mfcc=N_MFCC, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH
    )

    # 4. Analyze the Result
    print("\n--- MFCC RESULTS ---")
    print(f"MFCC Matrix Shape: {mfcc.shape}")
    print("   (Rows = n_mfcc, Columns = Time Frames)")

    # 5. Visualize
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC Heat Map (Input to your CNN)')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()