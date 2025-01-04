import tensorflow as tf

# Path to your original .h5 model
MODEL_PATH = "best_student_model.h5"

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Convert the model to TensorFlow Lite with optimizations
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Apply optimizations
tflite_model = converter.convert()

# Save the quantized model
TFLITE_MODEL_PATH = "student_model_quantized.tflite"
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Quantized model saved at: {TFLITE_MODEL_PATH}")
