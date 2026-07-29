import tensorflow_datasets as tfds

print("Downloading Cats vs Dogs dataset...")

dataset, info = tfds.load(
    "cats_vs_dogs",
    with_info=True,
    as_supervised=True,
)

print("\nDataset downloaded successfully!")
print(info)