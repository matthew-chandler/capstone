# Python

The python/ directory serves as the "Golden Reference" for the machine learning pipeline of the voice-controlled smart home project. It contains scripts for training, testing, and verifying the Keyword Spotting (KWS) model. The main components are:

* `train.py`: This is the main script for training the KWS model. It uses the TensorFlow and Keras libraries to:
    * Load the Google Speech Commands dataset.
    * Preprocess the audio data by converting it to Mel-Frequency Cepstral Coefficients (MFCCs). The script performs this conversion using tf.signal functions.
    * Define and train a Convolutional Neural Network (CNN) to classify the audio into three categories: "Yes", "No", and "Unknown".
    * Save the trained model as kws_model.keras.

* `test.py`: This script evaluates the performance of the trained model (kws_model.keras) on the test set. It loads the model and the test data, applies the same preprocessing as the training script, and then calculates the accuracy.
    It also includes a "spot check" feature to manually inspect a few predictions.

* `test_mfcc.py`: This script is used for verifying and visualizing the MFCC extraction process. It takes a single audio file, calculates the MFCCs using the librosa library (a popular audio analysis library), and then displays the
    resulting MFCC matrix as a heatmap. This helps in visually inspecting the features that are fed into the neural network and serves as a reference for the tf.signal implementation in train.py.

* `find_labels.py`: This is a utility script that lists all the possible labels in the original Speech Commands dataset. This is useful for understanding the full range of available commands and for mapping them to the simplified
    "Yes", "No", and "Unknown" classes used in this project.

* `predict.py`: This script is used to make predictions on a single audio file. It loads the trained model (`kws_model.keras`) and preprocesses the audio file in the same way as the training script.
    It takes an audio file as a command-line argument and outputs the predicted class and the confidence. It can also save a spectrogram of the audio file.

    Usage (from the `python` directory):
    ```
    python3 predict.py <path_to_audio_file> [--save-spec]
    ```

* `python/audio`: This directory contains example audio files for testing the prediction script. All files are in 16kHz mono PCM `.wav` format.
    * `yes.wav`
    * `no.wav`
    * `unknown.wav`



# Conda Commands: #
* Creating the environment & installing all dependencies (do this once at the start):
```conda env create -f python/environment.yml```
* Activating/entering the environment:
```conda activate capstone```
* Exiting the environment:
```conda deactivate```
* Updating the environment (if you add more dependencies):
```conda env update --file python/environment.yml --prune```
* Updating the environment file to match the current environment (please avoid this, apparently it can mess up the environment file):
```conda env export > python/environment_updated.yml```