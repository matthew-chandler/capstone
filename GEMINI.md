## Gemini Context: Voice-Controlled Embedded Smart Home
## 1. Project Overview
3
This project is an offline, privacy-preserving voice assistant built on an STM32H7 microcontroller. The goal is to perform real-time Keyword Spotting (KWS) using MFCC feature extraction and a CNN to trigger local smart home actions without cloud reliance.
## 2. Repository Structure

    /python: Contains the Python code for the project. This includes four key parts:
        - A "Golden Reference" MFCC implementation in `test_mfcc.py` using the `librosa` library. This is used for validating the C implementation.
        - A complete training and evaluation pipeline in `train.py` and `test.py` that uses `tf.signal` for preprocessing and `tf.keras` to build and train the CNN.
        - A script for running inference on a single audio file in `predict.py`.

    /stm32: Contains the C implementation for the MCU. This handles DMA audio acquisition, real-time MFCCs, and X-Cube-AI inference.

## 3. Technical Constraints & Definitions

    Audio Input: 16kHz Mono PCM.

    Feature Extraction (Log-Mel-Spectrogram): The training pipeline uses `tf.signal` to create a log-mel-spectrogram.
        - Frame Length: 640 samples
        - Frame Step: 320 samples
        - FFT Length: 1024
        - Mel Filterbanks: 40 triangular filters, from 20Hz to 4kHz.
        - No DCT is applied.

    Neural Network:

        Architecture: Small CNN (as defined in `train.py`).

        Dataset: Google Speech Commands (mapped to 'Yes', 'No', 'Unknown').

        Optimization: Quantization (INT8) and pruning to fit memory.

    Target Hardware: STM32H7 (ARM Cortex-M7), using CMSIS-DSP and X-Cube-AI.

## 4. Development Workflow

    Stage 1 (Python): Use `test_mfcc.py` as the "Golden Reference" for the `librosa` based MFCC implementation.

    Stage 2 (Python): Train the CNN in Keras/TensorFlow using `train.py` which saves the model as a `.keras` file.

    Stage 3 (C): Replicate the Python MFCC math in C and verify bit-accuracy against the "Golden Reference".

    Stage 4 (C): Use X-Cube-AI to port the model and run real-time inference.

## 5. Prompting Instructions for Gemini

When generating code for this repo, please:

    Note: There are two different MFCC implementations in the `/python` directory.
        - `test_mfcc.py` uses `librosa` and is considered the "Golden Reference".
        - `train.py`, `test.py`, and `predict.py` use `tf.signal` for the training and evaluation pipeline.

    Ensure Python signal processing uses librosa or numpy for the reference model.

    Prioritize memory efficiency (low parameter count) for all CNN architectures.

    When writing C code, assume the use of CMSIS-DSP functions (e.g., arm_rfft_fast_f32).

## Next Step