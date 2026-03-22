# Voice-Controlled Embedded Smart Home

This project is an offline, privacy-preserving voice assistant built on an STM32H7 microcontroller. The goal is to perform real-time Keyword Spotting (KWS) using MFCC feature extraction and a Convolutional Neural Network (CNN) to trigger local smart home actions without cloud reliance.

## Directory Structure
- `python/`: Contains the Python code for training the Keyword Spotting (KWS) model and validating the Machine Learning pipeline.
- `stm/`: Contains the C implementation for the STM32 MCU, handling real-time continuous audio acquisition, DSP feature extraction, and X-Cube-AI inference.

---

## Python Source Files (`python/`)

The `python/` directory serves as the "Golden Reference" for the machine learning pipeline. It contains scripts for training, testing, visualizing, and verifying the KWS model.

### Core Pipeline
* `train.py`: The main script for training the KWS model using TensorFlow/Keras. It loads the Google Speech Commands dataset, preprocesses audio into log-mel-spectrograms (MFCCs) using `tf.signal`, defines the CNN architecture, and saves the trained model (`kws_model.keras` and `kws_model.h5`).
* `test.py`: Evaluates the performance of the trained model on the test dataset. Prints the accuracy and performs spot-checks on sample predictions.
* `predict.py`: Makes predictions on a single audio file. Loads the trained model, preprocesses the input identically to `train.py`, and outputs the predicted class with confidence. Also supports saving a spectrogram.
* `test_mfcc.py`: Extracts and visualizes MFCCs for a single audio file using the `librosa` library. This serves as the "Golden Reference" to validate the `tf.signal` and CMSIS-DSP equivalent implementations.
* `find_labels.py`: Utility script that lists all possible labels in the original Speech Commands dataset, mapping them down to the simplified "Yes", "No", and "Unknown" classes used in this project.

### Utilities and STM32 Export Tools
* `visualize_model.py`: Generates a Visualkeras architectural diagram of the KWS CNN. Outputs `model_architecture.png`.
* `export_c_constants.py`: Generates `mel_constants.h` containing the exact Hann window values and Mel filterbank matrix initialized in Python, so they can be seamlessly injected into the C code.
* `export_model.py`: Converts the Keras model (`kws_model.h5`) into a TensorFlow Lite model (`kws_model.tflite`) for embedded deployment optimization.
* `verify_dsp_stm32.py`: Generates the Python ground truth of the MFCC transformations over a dummy linear ramp signal. This gives us exact float values to compare against when verifying the STM32's DSP math bit-accuracy.
* `get_custom_samples.py`: Takes the `yes.wav`, `no.wav`, and `unknown.wav` testing files and converts them into C arrays in a single header file (`test_audio.h`) to reliably simulate identical microphone input on the MCU.
* `get_one_sample.py`: Extracts the first "Yes" sample directly from the real dataset and similarly generates a `test_audio.h` file containing its raw audio format.

---

## STM32 Source Files (`stm/Core/Src/`)

The `stm/` directory holds the embedded C code for the project, configured for STM32CubeIDE.

* `main.c`: Orchestrates the system's entire execution. It initializes the peripherals, configures a Timer-triggered ADC with DMA for endless background microphone recording, and systematically stitches audio blocks as they arrive. It then feeds the audio into the DSP and AI handlers and toggles LEDs based on the prediction.
* `dsp_pipeline.c`: Implements the identical, bit-accurate MFCC feature extraction pipeline strictly using CMSIS-DSP functions. It loops over the audio buffer sliding window, applies the Hann window, executes the Real Fast Fourier Transform (RFFT), applies the Mel filterbank matrix, computes the natural logarithm, and formats an output tensor matched for the CNN.
* `cnn.c`: Handles the initialization and execution of the neural network via the ST X-CUBE-AI library. It directs the network to read the tensor created by `dsp_pipeline.c`, runs the inference layer-by-layer, and extracts the float percentage confidence scores.

---

## Usage Instructions

### 1. Setting up the Conda Environment
To smoothly run all the Python scripts, a Conda environment will manage the exact dependencies (like TensorFlow, Librosa, and matplotlib).
* **Create the environment & install dependencies** (do this once at the start):
  ```bash
  conda env create -f python/environment.yml
  ```
* **Activate the environment**:
  ```bash
  conda activate capstone
  ```
* **Deactivate the environment** (when done):
  ```bash
  conda deactivate
  ```
* **Update the environment** (if `environment.yml` changes):
  ```bash
  conda env update --file python/environment.yml --prune
  ```

### 2. Training and Testing the Model
From within the `python/` directory, while your `capstone` Conda environment is active:
1. **Train the model**:
   ```bash
   python test.py
   python train.py
   ```
   *This will download the dataset (if missing), train the multi-layer CNN, and save the weights/model file.*
2. **Test the model's accuracy on the unknown data split**:
   ```bash
   python test.py
   ```
3. **Run a single prediction on an individual `.wav` file**:
   ```bash
   python predict.py audio/yes.wav --save-spec
   ```

### 3. Setting up X-CUBE-AI in STM32CubeIDE
To deploy the successfully trained model onto the STM32 microcontroller:
1. Open the project's `.ioc` configuration file in **STM32CubeIDE**.
2. Assuming `X-CUBE-AI` is already downloaded to your machine, open **Software Packs** -> **Select Components**. Ensure **STMicroelectronics.X-CUBE-AI** is selected and added to your project.
3. On the left sidebar under Categories, navigate to **Software Packs** -> **STMicroelectronics.X-CUBE-AI** -> **Artificial Intelligence**. Check the box to enable the AI core.
4. In the main configuration panel that appears:
   - Next to Network 1, click **Add Network**.
   - Provide a Name, and select **Keras** (or **TFLite**) in the dropdown based on your exported model.
   - Click **Browse** and correctly locate either `kws_model.h5` or `kws_model.tflite` from the `python/` directory.
   - Click **Analyze**. This step will parse the neural network topology, optimize it, and report the required Flash/RAM. Ensure the sizes comfortably fit inside the STM32H7 limits.
5. Click **Generate Code** (or `Ctrl+S`) to apply your changes.
6. Finally, build and flash the project onto your STM32!