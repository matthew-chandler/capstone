import tensorflow_datasets as tfds

# Load the builder info (doesn't download the whole dataset again)
builder = tfds.builder('speech_commands')
class_names = builder.info.features['label'].names

print(f"Total Labels: {len(class_names)}")
print("-" * 30)
for i, name in enumerate(class_names):
    print(f"{i}: {name}")