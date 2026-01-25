"""
Trenowanie klasyfikatora ubrań na zbiorze Fashion-MNIST.

Przykładowe użycie:
    MLP
    python train_fashion_mnist.py --model-type dense --epochs 10 --batch-size 128 --output-dir outputs_dense

    CNN
    python train_fashion_mnist.py --model-type cnn --epochs 10 --batch-size 128 --output-dir outputs_cnn

Program:
- wczytuje zbiór Fashion-MNIST z tf.keras.datasets
- trenuje model (dense albo cnn)
- zapisuje model do pliku .keras
- zapisuje metryki (loss, accuracy) do JSON
- zapisuje macierz pomyłek do CSV

Architektury są przygotowane tak, aby można je było łatwo wykorzystać z Keras Tunerem
(popatrz na funkcje: build_dense_model i build_cnn_model – przyjmują parametr hp).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Any, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_fashion_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Wczytuje zbiór Fashion-MNIST z tf.keras.datasets.

    Zwraca:
        (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizacja i przygotowanie danych wejściowych.

    - skaluje piksele do [0, 1]
    - dodaje kanał (28, 28) -> (28, 28, 1)
    """
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, x_test

def build_dense_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    hp: Optional[Any] = None,
) -> tf.keras.Model:
    """
    Buduje sieć w pełni połączoną (MLP).

    Parametr `hp` jest opcjonalnym obiektem HyperParameters z Keras Tunera.
    Jeśli jest podany, hiperparametry (liczba neuronów, dropout, itd.) są
    pobierane z `hp`. Jeśli nie – używamy stałych wartości domyślnych.
    """
    if hp is not None:
        units_1 = hp.Int("dense_units_1", min_value=128, max_value=512, step=64)
        units_2 = hp.Int("dense_units_2", min_value=64, max_value=512, step=64)
        dropout_rate = hp.Float("dense_dropout", min_value=0.1, max_value=0.5, step=0.1)
    else:
        units_1 = 256
        units_2 = 128
        dropout_rate = 0.3

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(units_1, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units_2, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fashion_mnist_dense")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    hp: Optional[Any] = None,
) -> tf.keras.Model:
    """
    Buduje sieć splotową (CNN).

    Parametr `hp` jest opcjonalnym obiektem HyperParameters z Keras Tunera.
    """
    if hp is not None:
        filters_1 = hp.Int("conv_filters_1", min_value=16, max_value=64, step=16)
        filters_2 = hp.Int("conv_filters_2", min_value=32, max_value=128, step=32)
        dense_units = hp.Int("cnn_dense_units", min_value=64, max_value=256, step=64)
        dropout_rate = hp.Float("cnn_dropout", min_value=0.1, max_value=0.5, step=0.1)
    else:
        filters_1 = 32
        filters_2 = 64
        dense_units = 128
        dropout_rate = 0.3

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters_1, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(filters_2, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fashion_mnist_cnn")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model(
    model_type: str,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    hp: Optional[Any] = None,
) -> tf.keras.Model:
    """
    Zwraca model w zależności od typu architektury.
    """
    model_type = model_type.lower()
    if model_type == "dense":
        return build_dense_model(input_shape, num_classes, hp)
    elif model_type == "cnn":
        return build_cnn_model(input_shape, num_classes, hp)
    else:
        raise ValueError(f"Nieobsługiwany typ modelu: {model_type}. Użyj 'dense' lub 'cnn'.")

def train_and_evaluate(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    epochs: int,
) -> dict:
    """
    Trenuje model i zwraca słownik z podstawowymi metrykami:
    - test_loss
    - test_accuracy
    oraz macierz pomyłek (jako numpy array w polu 'confusion_matrix').
    """
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_prob = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    cm = tf.math.confusion_matrix(
        y_test,
        y_pred,
        num_classes=len(CLASS_NAMES),
        dtype=tf.int32,
    ).numpy()

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
        "confusion_matrix": cm,
    }

    return metrics


def save_model_and_metrics(
    model: tf.keras.Model,
    metrics: dict,
    output_dir: Path,
    model_type: str,
) -> None:
    """
    Zapisuje:
    - model (.keras)
    - podstawowe metryki (loss, accuracy, historia) do JSON
    - macierz pomyłek jako GRAFIKĘ (PNG)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"fashion_mnist_{model_type}.keras"
    model.save(model_path)
    print(f"Model zapisany do: {model_path}")

    metrics_to_save = {
        "test_loss": metrics["test_loss"],
        "test_accuracy": metrics["test_accuracy"],
        "history": metrics["history"],
    }

    metrics_path = output_dir / f"metrics_{model_type}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Metryki zapisane do: {metrics_path}")

    cm = metrics["confusion_matrix"]
    cm_plot_path = output_dir / f"confusion_matrix_{model_type}.png"
    save_confusion_matrix_plot(cm, cm_plot_path, CLASS_NAMES)
    print(f"Macierz pomyłek zapisana jako obrazek do: {cm_plot_path}")

    print("Kolejność klas w macierzy pomyłek (wiersze/kolumny):")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"{idx}: {name}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trenowanie klasyfikatora ubrań (Fashion-MNIST)."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="dense",
        choices=["dense", "cnn"],
        help="Typ modelu: 'dense' (MLP) lub 'cnn' (sieć splotowa).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Liczba epok trenowania.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Rozmiar batcha.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Katalog, do którego zostanie zapisany model i metryki.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    x_train, x_test = preprocess_data(x_train, x_test)

    input_shape = x_train.shape[1:]
    num_classes = len(CLASS_NAMES)

    print(f"Rozmiar zbioru treningowego: {x_train.shape[0]}")
    print(f"Rozmiar zbioru testowego: {x_test.shape[0]}")
    print(f"Kształt wejścia: {input_shape}")
    print(f"Liczba klas: {num_classes}")

    model = get_model(
        model_type=args.model_type,
        input_shape=input_shape,
        num_classes=num_classes,
        hp=None,
    )

    model.summary()

    metrics = train_and_evaluate(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")

    output_dir = Path(args.output_dir)
    save_model_and_metrics(
        model=model,
        metrics=metrics,
        output_dir=output_dir,
        model_type=args.model_type,
    )

def save_confusion_matrix_plot(cm, output_path, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    main()
