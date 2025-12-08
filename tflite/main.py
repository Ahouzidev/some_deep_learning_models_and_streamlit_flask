import tensorflow as tf

# Load your existing Keras model
model = tf.keras.models.load_model("fruits_cnn.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("fruits_cnn.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved!")
