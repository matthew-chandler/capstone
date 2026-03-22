"""
Generates a Visualkeras diagram of the KWS CNN architecture.
Outputs: model_architecture.png

Includes a compatibility patch for Keras 3 + visualkeras.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
import tensorflow as tf
import keras

# --- Rebuild the model architecture (no training needed) ---
# Input shape: (49 time frames, 40 mel bins, 1 channel)
input_shape = (49, 40, 1)

model = keras.models.Sequential([
    # Preprocessing
    keras.layers.Resizing(32, 32, input_shape=input_shape),
    keras.layers.Normalization(),

    # First Convolution Block
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),

    # Second Convolution Block
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),

    # Classifier Head
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),

    # Output Layer
    keras.layers.Dense(3, activation='softmax')
])

# Build the model so shapes are computed
model.build(input_shape=(None,) + input_shape)
model.summary()

# --- Keras 3 compatibility patch for visualkeras ---
# visualkeras expects layer.output_shape (Keras 2 API),
# but Keras 3 uses layer.output.shape instead.
for layer in model.layers:
    if not hasattr(layer, 'output_shape') or not callable(getattr(layer, 'output_shape', None)):
        try:
            shape = layer.output.shape
            # Convert to tuple format that visualkeras expects
            layer.output_shape = shape
        except Exception:
            pass

import visualkeras
from PIL import Image, ImageDraw, ImageFont

# --- Color map matching visualkeras defaults for each layer type ---
# visualkeras assigns colors by layer class; we replicate them here for the legend.
layer_color_map = {
    'Resizing':       '#f0c571',  # gold
    'Normalization':  '#e8527a',  # pink
    'Conv2D':         '#00b4d8',  # cyan
    'MaxPooling2D':   '#0d3b66',  # dark teal
    'Dropout':        '#9b5de5',  # purple
    'Flatten':        '#f4a7bb',  # light pink
    'Dense':          '#f28c5a',  # orange
}

# --- Generate Visualkeras diagram (no legend, thinner layers) ---
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_architecture.png")

arch_img = visualkeras.layered_view(
    model,
    legend=False,
    spacing=25,
    scale_xy=2,
    scale_z=0.5,       # thinner layers
    max_xy=300,
    max_z=80,
    draw_volume=True
)

# Rotate architecture 90° clockwise (input at top, output at bottom)
arch_img = arch_img.rotate(-90, expand=True)

# --- Draw a custom horizontal legend to the right ---
padding = 30
swatch_size = 18
line_height = 28
legend_items = list(layer_color_map.items())

# Try to load a clean font, fall back to default
try:
    font = ImageFont.truetype("arial.ttf", 16)
except OSError:
    font = ImageFont.load_default()

# Measure legend width
max_text_w = max(font.getbbox(name)[2] for name, _ in legend_items)
legend_w = swatch_size + 10 + max_text_w + padding * 2
legend_h = len(legend_items) * line_height + padding * 2

# Compose final image: architecture on the left, legend on the right
final_w = arch_img.width + legend_w + padding
final_h = max(arch_img.height, legend_h) + padding * 2
final = Image.new('RGBA', (final_w, final_h), (255, 255, 255, 255))

# Paste architecture (centered vertically)
arch_y = (final_h - arch_img.height) // 2
final.paste(arch_img, (padding, arch_y), arch_img if arch_img.mode == 'RGBA' else None)

# Draw legend (centered vertically on the right side)
draw = ImageDraw.Draw(final)
legend_x = arch_img.width + padding * 2
legend_y = (final_h - legend_h) // 2

for i, (name, color) in enumerate(legend_items):
    y = legend_y + padding + i * line_height
    # Color swatch
    draw.rectangle([legend_x, y, legend_x + swatch_size, y + swatch_size],
                   fill=color, outline='#333333')
    # Label
    draw.text((legend_x + swatch_size + 10, y - 2), name, fill='#222222', font=font)

final.save(output_path)

print(f"\n✅ Architecture diagram saved to: {output_path}")
