import os
import numpy as np
import pandas as pd
import tensorflow as tf
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class Config:
    csv_path: str = "penguins_size.csv"
    model_path: str = "penguins_sex.keras"
    feature_cols: tuple = ("culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g")
    target_col: str = "sex"
    allowed_targets: tuple = ("MALE", "FEMALE")

    test_size: float = 0.2
    random_state: int = 42

    epochs: int = 30
    batch_size: int = 32
    lr: float = 0.01
    val_split: float = 0.2

def load_and_prepare_data(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(cfg.csv_path)

    needed = list(cfg.feature_cols) + [cfg.target_col]
    df = df[needed].dropna()
    df = df[df[cfg.target_col].isin(cfg.allowed_targets)]

    X = df[list(cfg.feature_cols)].astype("float32").values
    y = (df[cfg.target_col] == "MALE").astype("int32").values  # 1=MALE, 0=FEMALE
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # stratify tylko jeśli obie klasy mają >=2 próbki
    stratify_opt = y if pd.Series(y).value_counts().min() >= 2 else None

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_opt
    )

def build_model(input_dim: int, X_train: np.ndarray, lr: float) -> tf.keras.Model:
    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(X_train)

    inputs = tf.keras.Input(shape=(input_dim,))
    x = norm(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    val_split: float,
    verbose: int = 1
) -> tf.keras.callbacks.History:
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=verbose
    )
    return history


def load_or_train_and_save(
    cfg: Config,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> tuple[tf.keras.Model, tf.keras.callbacks.History | None]:
    if os.path.isfile(cfg.model_path):
        model = tf.keras.models.load_model(cfg.model_path)
        print(f"Existing model loaded: {cfg.model_path}")
        return model, None

    model = build_model(input_dim=X_train.shape[1], X_train=X_train, lr=cfg.lr)
    history = train_model(
        model,
        X_train, y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        verbose=1
    )
    model.save(cfg.model_path)
    print(f"New model saved to: {cfg.model_path}")
    return model, history


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return loss, acc

def main():
    cfg = Config()

    X, y = load_and_prepare_data(cfg)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    model, history = load_or_train_and_save(cfg, X_train, y_train)

    loss, acc = evaluate_model(model, X_test, y_test)
    print(f"Test accuracy: {acc:.4f}, loss: {loss:.4f}")


if __name__ == "__main__":
    main()
