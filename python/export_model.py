import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

print("Loading Keras model...")
model = tf.keras.models.load_model('kws_model.h5') # Load the .h5 file

print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
output_name = 'kws_model.tflite'
with open(output_name, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Success! Saved as '{output_name}'")