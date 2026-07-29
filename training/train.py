import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

MODEL_DIR = Path("models/final")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load Dataset
# ----------------------------
print("\nLoading dataset...")

(ds_train, ds_test), ds_info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

print("Dataset loaded successfully.")

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


train_ds = (
    ds_train
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    ds_test
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ----------------------------
# Build Model
# ----------------------------
print("\nBuilding MobileNetV2 model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ----------------------------
# Train
# ----------------------------
print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
)

# ----------------------------
# Evaluate
# ----------------------------
loss, accuracy = model.evaluate(test_ds)

print(f"\nValidation Accuracy: {accuracy:.4f}")

# ----------------------------
# Save Model
# ----------------------------
model_path = MODEL_DIR / "cat_dog_classifier.keras"

model.save(model_path)

print("\nModel saved successfully.")
print(model_path)