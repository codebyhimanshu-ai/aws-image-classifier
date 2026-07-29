import numpy as np
import tensorflow as tf
from PIL import Image

from configs.config import MODEL_PATH, IMAGE_SIZE

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


def predict_image(image: Image.Image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0][0]

    if prediction >= 0.5:
        label = "Dog"
        confidence = prediction
    else:
        label = "Cat"
        confidence = 1 - prediction

    return {
        "prediction": label,
        "confidence": round(float(confidence * 100), 2)
    }