"""
Klasyfikacja pojedynczego obrazka przy użyciu wytrenowanego modelu Fashion-MNIST.

Przykładowe użycie:
    python predict_fashion_mnist.py dress_test.png
    python predict_fashion_mnist.py --model-path outputs_dense/fashion_mnist_dense.keras image.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    """
    Wczytuje obrazek z dysku i przetwarza go tak, aby pasował
    do modelu trenowanego na Fashion-MNIST.

    Kroki:
    - wczytanie obrazka,
    - konwersja do odcieni szarości,
    - skalowanie do 28x28,
    - zastosowanie negatywu (inwersja),
    - normalizacja do [0, 1],
    - dodanie wymiaru batcha i kanału: (1, 28, 28, 1).
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Plik nie istnieje: {image_path}")

    img = Image.open(image_path)

    img = img.convert("L")

    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    img = ImageOps.invert(img)

    img_array = np.array(img).astype("float32")

    img_array /= 255.0

    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def load_model(model_path: Path) -> tf.keras.Model:
    """
    Wczytuje zapisany model .keras.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model nie istnieje: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def predict_image(
    model: tf.keras.Model,
    image_array: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    """
    Wykonuje predykcję na pojedynczym obrazku.

    Zwraca:
        predicted_class_idx: indeks klasy (0-9)
        confidence: pewność predykcji dla tej klasy (0-1)
        probs: wektor prawdopodobieństw dla wszystkich klas
    """
    probs = model.predict(image_array, verbose=0)[0]
    predicted_class_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_class_idx])
    return predicted_class_idx, confidence, probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Klasyfikacja obrazka przy użyciu modelu Fashion-MNIST."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Ścieżka do pliku z obrazkiem do klasyfikacji.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs_dense/fashion_mnist_dense.keras",
        help=(
            "Ścieżka do wytrenowanego modelu (.keras). "
            "Domyślnie: outputs_dense/fashion_mnist_dense.keras"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    model_path = Path(args.model_path)

    print(f"Używany model:   {model_path}")
    print(f"Klasyfikowany plik: {image_path}")

    img_array = load_and_preprocess_image(image_path)

    model = load_model(model_path)

    predicted_idx, confidence, probs = predict_image(model, img_array)

    predicted_label = CLASS_NAMES[predicted_idx]

    print("\nWYNIK KLASYFIKACJI")
    print(f"Przewidziana klasa: {predicted_label} (id={predicted_idx})")
    print(f"Pewność predykcji:  {confidence * 100:.2f}%")

if __name__ == "__main__":
    main()
