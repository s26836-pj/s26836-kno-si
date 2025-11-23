from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os, shutil, json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, AlphaDropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

SEED = 42
TEST_SIZE = 0.2
VAL_SPLIT = 0.2

# Model A (ReLU + He + Adam)
EPOCHS_A = 120
BATCH_A  = 16
LR_A     = 1e-3

# Model B (SELU + LeCun + SGD)
EPOCHS_B = 160
BATCH_B  = 16
LR_B     = 1e-2

USE_TB = False
PREDICT_ONLY = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

def ensure_dir_is_directory(path: Path):
    if path.exists() and not path.is_dir():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)

def force_clean_dir(p: Path):
    if p.exists():
        if p.is_file():
            print(f"[LOGDIR] {p} istnieje jako PLIK -> usuwam")
            p.unlink()
        else:
            print(f"[LOGDIR] {p} istnieje jako katalog -> czyszczę")
            shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    print(f"[LOGDIR] OK: {p} utworzony jako katalog")

if __name__ == "__main__":
    import argparse
    _base = argparse.ArgumentParser(add_help=False)
    _base.add_argument("--predict-only", action="store_true", help="Pomiń trening i wykonaj tylko predykcję (wymaga zapisanych modelu i scaler'a).")
    _base.add_argument("--use-tb", action="store_true", help="Włącz TensorBoard callbacks.")
    _base.add_argument("--no-plots", action="store_true", help="Nie pokazuj wykresów matplotlib.")
    _known, _remaining = _base.parse_known_args()

    PREDICT_ONLY = _known.predict_only
    USE_TB = _known.use_tb
    NO_PLOTS = _known.no_plots
else:
    _remaining = []
    NO_PLOTS = False

HERE = Path(__file__).resolve().parent
LOGROOT = (HERE / "runs_tf" / f"wine_{datetime.now().strftime('%Y%m%d-%H%M%S')}").resolve()
MODELS_DIR = HERE / "models"

CSV_CANDIDATES = [
    HERE / "wine_shuffled.csv",
    HERE / "data" / "processed" / "wine.csv",
]
csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
if csv_path is None:
    raise FileNotFoundError(f"Nie znaleziono wine.csv. Sprawdziłem: {CSV_CANDIDATES}")

print(f"[INFO] Używam pliku: {csv_path}")

COLUMN_NAMES = [
    "class","alcohol","malic_acid","ash","alcalinity_of_ash","magnesium",
    "total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins",
    "color_intensity","hue","od280/od315_of_diluted_wines","proline"
]

df_try = pd.read_csv(csv_path, nrows=1)
if "class" in df_try.columns:
    df = pd.read_csv(csv_path)
    if len(df.columns) != 14 or df.columns[0] != "class":
        df.columns = COLUMN_NAMES
else:
    df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)

print(f"[INFO] shape: {df.shape}")
print(f"[INFO] klasy: {sorted(df['class'].unique())}")

X = df.drop(columns=["class"]).to_numpy(dtype=np.float32)
y_raw = df["class"].to_numpy(dtype=np.int64)
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X))
X = X[perm]
y_raw = y_raw[perm]

num_classes = 3
y0 = y_raw - 1
assert set(np.unique(y0)) == {0,1,2}, "Etykiety powinny być w {1,2,3}."
Y = np.eye(num_classes, dtype=np.float32)[y0]

def stratified_split(n_classes: int, y_raw_1based: np.ndarray, test_size: float, seed: int):
    rs = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y_raw_1based):
        idx = np.where(y_raw_1based == c)[0]
        rs.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    rs.shuffle(train_idx)
    rs.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)

train_idx, test_idx = stratified_split(num_classes, y_raw, TEST_SIZE, SEED)
X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]
print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

mean = X_train.mean(axis=0, keepdims=True)
std  = X_train.std(axis=0,  keepdims=True)
std[std == 0.0] = 1.0
X_train_std = (X_train - mean) / std
X_test_std  = (X_test  - mean) / std

def build_model_a(input_dim: int, n_classes: int) -> Sequential:
    model = Sequential(
        name="WineNet_A_ReLU_He",
        layers=[
            Input(shape=(input_dim,), name="input"),
            Dense(32, activation="relu", kernel_initializer="he_normal", name="dense_relu_32"),
            Dense(16, activation="relu", kernel_initializer="he_normal", name="dense_relu_16"),
            Dense(n_classes, activation="softmax", name="output_softmax"),
        ],
    )
    model.compile(
        optimizer=Adam(learning_rate=LR_A),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_model_b(input_dim: int, n_classes: int) -> Sequential:
    model = Sequential(
        name="WineNet_B_SELU_LeCun",
        layers=[
            Input(shape=(input_dim,), name="input"),
            Dense(64, activation="selu", kernel_initializer="lecun_normal", name="dense_selu_64"),
            AlphaDropout(0.1, name="alpha_dropout_01"),
            Dense(32, activation="selu", kernel_initializer="lecun_normal", name="dense_selu_32"),
            Dense(n_classes, activation="softmax", name="output_softmax"),
        ],
    )
    model.compile(
        optimizer=SGD(learning_rate=LR_B, momentum=0.9, nesterov=True),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model_a = build_model_a(X_train_std.shape[1], num_classes)
model_b = build_model_b(X_train_std.shape[1], num_classes)

model_a.summary()
model_b.summary()

LOG_A = LOGROOT / "model_A"
LOG_B = LOGROOT / "model_B"

if USE_TB:
    force_clean_dir(LOGROOT)
    force_clean_dir(LOG_A)
    force_clean_dir(LOG_B)
else:
    ensure_dir_is_directory(LOGROOT)
    ensure_dir_is_directory(LOG_A)
    ensure_dir_is_directory(LOG_B)

cb_a = [EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True)]
cb_b = [EarlyStopping(monitor="val_accuracy", patience=30, restore_best_weights=True)]
if USE_TB:
    cb_a.append(TensorBoard(LOG_A.as_posix(), histogram_freq=1))
    cb_b.append(TensorBoard(LOG_B.as_posix(), histogram_freq=1))

hist_a = hist_b = None

if not PREDICT_ONLY:
    print(f"\n[TRAIN] Model A  | epochs={EPOCHS_A}, lr={LR_A}, batch_size={BATCH_A}")
    hist_a = model_a.fit(
        X_train_std, Y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS_A,
        batch_size=BATCH_A,
        callbacks=cb_a,
        verbose=0
    )

    print(f"\n[TRAIN] Model B  | epochs={EPOCHS_B}, lr={LR_B}, batch_size={BATCH_B}")
    hist_b = model_b.fit(
        X_train_std, Y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS_B,
        batch_size=BATCH_B,
        callbacks=cb_b,
        verbose=0
    )
else:
    print("\n[MODE] PREDICT-ONLY: pomijam trening.")


if not PREDICT_ONLY:
    test_loss_a, test_acc_a = model_a.evaluate(X_test_std, Y_test, verbose=0)
    test_loss_b, test_acc_b = model_b.evaluate(X_test_std, Y_test, verbose=0)

    MODELS_DIR.mkdir(exist_ok=True)
    model_a.save(MODELS_DIR / "model_A.keras")
    model_b.save(MODELS_DIR / "model_B.keras")
    print(f"[SAVE] Wagi: {MODELS_DIR/'model_A.keras'}, {MODELS_DIR/'model_B.keras'}")

    def confusion_matrix_np(y_true_oh: np.ndarray, y_pred_oh: np.ndarray, n_classes: int) -> np.ndarray:
        t = y_true_oh.argmax(axis=1)
        p = y_pred_oh.argmax(axis=1)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for ti, pi in zip(t, p):
            cm[ti, pi] += 1
        return cm

    Y_test_pred_a = model_a.predict(X_test_std, verbose=0)
    Y_test_pred_b = model_b.predict(X_test_std, verbose=0)
    cm_a = confusion_matrix_np(Y_test, Y_test_pred_a, num_classes)
    cm_b = confusion_matrix_np(Y_test, Y_test_pred_b, num_classes)

    FIGS_DIR = HERE / "figs"
    FIGS_DIR.mkdir(exist_ok=True)

    def plot_and_save_cm(cm: np.ndarray, title: str, path: Path):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        plt.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    plot_and_save_cm(cm_a, "Confusion Matrix — Model A", FIGS_DIR / "cm_model_A.png")
    plot_and_save_cm(cm_b, "Confusion Matrix — Model B", FIGS_DIR / "cm_model_B.png")
    print(f"[SAVE] CM: {FIGS_DIR/'cm_model_A.png'}, {FIGS_DIR/'cm_model_B.png'}")

    results = {
        "seed": SEED,
        "split": {"test_size": TEST_SIZE, "val_split_from_train": VAL_SPLIT},
        "model_A": {"epochs": EPOCHS_A, "batch_size": BATCH_A, "learning_rate": LR_A,
                    "test_loss": float(test_loss_a), "test_acc": float(test_acc_a)},
        "model_B": {"epochs": EPOCHS_B, "batch_size": BATCH_B, "learning_rate": LR_B,
                    "test_loss": float(test_loss_b), "test_acc": float(test_acc_b)},
    }
    with open(HERE / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Wyniki: {HERE/'results.json'}")

    np.savez(HERE / "scaler_train_stats.npz", mean=mean, std=std)

    def plot_history(h, title: str):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(h.history["loss"], label="train")
        axes[0].plot(h.history["val_loss"], label="val")
        axes[0].set_title(f"{title} — Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()

        axes[1].plot(h.history["accuracy"], label="train")
        axes[1].plot(h.history["val_accuracy"], label="val")
        axes[1].set_title(f"{title} — Accuracy")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
        plt.tight_layout()
        plt.show()

    print(f"Model A — loss: {test_loss_a:.4f} | acc: {test_acc_a:.4f} | epochs: {EPOCHS_A} | lr: {LR_A} | batch: {BATCH_A}")
    print(f"Model B — loss: {test_loss_b:.4f} | acc: {test_acc_b:.4f} | epochs: {EPOCHS_B} | lr: {LR_B} | batch: {BATCH_B}")

    if not NO_PLOTS:
        plot_history(hist_a, "Model A (ReLU+He+Adam)")
        plot_history(hist_b, "Model B (SELU+LeCun+SGD)")

    if USE_TB:
        print(f"\n[TENSORBOARD] uruchom w terminalu:\n  tensorboard --logdir {LOGROOT.as_posix()}\n")

def run_cli_prediction(args, feature_names):
    """
    Obsługa predykcji z linii komend:
    - oczekuje 13 cech wina jako parametrów CLI,
    - korzysta z zapisanych: scaler_train_stats.npz, model_A.keras, model_B.keras (jeśli istnieje).
    """
    cli_to_feat = {feat.replace("/", "_"): feat for feat in feature_names}

    any_feature = any(getattr(args, cli) is not None for cli in cli_to_feat.keys())
    if not any_feature:
        if PREDICT_ONLY:
            print("\n[INFO] PREDICT-ONLY bez podanych cech. "
                  "Podaj 13 cech jako --alcohol ... --proline, itd.")
        else:
            print("\n[INFO] Nie podano cech wina — zakończono trening/ewaluację, predykcję pomijam.")
        return

    scaler_stats_path = HERE / "scaler_train_stats.npz"
    if not scaler_stats_path.exists():
        raise FileNotFoundError(
            f"Brak statystyk standaryzacji: {scaler_stats_path} "
            f"(uruchom trening albo dostarcz plik)."
        )

    scaler_stats = np.load(scaler_stats_path)
    mean, std = scaler_stats["mean"], scaler_stats["std"]

    row, missing = [], []
    for feat in feature_names:
        cli = feat.replace("/", "_")
        val = getattr(args, cli)
        if val is None:
            missing.append(cli)
            row.append(0.0)
        else:
            row.append(float(val))

    if missing:
        print(f"[WARN] Nie podano wartości dla: {', '.join(missing)}. Używam 0.0 dla braków.")

    x = np.array([row], dtype=np.float32)
    x_std = (x - mean) / std

    model_a_path = MODELS_DIR / "model_A.keras"
    if not model_a_path.exists():
        raise FileNotFoundError(
            f"Brak modelu A: {model_a_path} (uruchom trening albo dostarcz plik)."
        )

    model_a = tf.keras.models.load_model(model_a_path)
    probs_a = model_a.predict(x_std, verbose=0)
    predicted_class_a = int(np.argmax(probs_a) + 1)

    print("\n" + "=" * 40)
    print("WYNIK KLASYFIKACJI (Model A)")
    print(f"Predykowana klasa wina: {predicted_class_a}  (1..3)")
    print("Prawdopodobieństwa [class1, class2, class3]:", np.round(probs_a[0], 3))
    print("=" * 40)

    model_b_path = MODELS_DIR / "model_B.keras"
    if model_b_path.exists():
        model_b = tf.keras.models.load_model(model_b_path)

        probs_b = model_b.predict(x_std, verbose=0)
        predicted_class_b = int(np.argmax(probs_b) + 1)

        print("\n" + "=" * 40)
        print("WYNIK KLASYFIKACJI (Model B)")
        print(f"Predykowana klasa wina: {predicted_class_b}  (1..3)")
        print("Prawdopodobieństwa [class1, class2, class3]:", np.round(probs_b[0], 3))
        print("=" * 40)
    else:
        print(
            f"\n[INFO] Pomijam predykcję Modelu B: Nie znaleziono pliku {model_b_path.name} "
            f"(wymagany jest trening)."
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Klasyfikacja wina (Model A/B) – podaj 13 cech jako argumenty CLI."
    )

    feature_names = COLUMN_NAMES[1:]  # pomijamy 'class'
    for feat in feature_names:
        cli = feat.replace("/", "_")
        parser.add_argument(
            f"--{cli}",
            type=float,
            required=False,
            help=f"Wartość cechy: {feat}",
        )

    # Używamy _remaining z wcześniejszego parse_known_args()
    args = parser.parse_args(_remaining)

    run_cli_prediction(args, feature_names)


#Podzilic rowniez na funkcje wystarcza.