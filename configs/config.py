from pathlib import Path

# ===========================
# Project Paths
# ===========================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final" / "cat_dog_classifier.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Cat",
    "Dog"
]