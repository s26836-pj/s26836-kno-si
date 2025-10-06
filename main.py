import argparse, os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Number prediction")
parser.add_argument("--model", default="model.keras", help="model name")
parser.add_argument("image", nargs="?", help="image path")
args = parser.parse_args()

if os.path.isfile(args.model):
    model = tf.keras.models.load_model(args.model)
else:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test)
    model.save(args.model)

    if history is not None:
        hist = history.history
        # Loss
        plt.figure()
        plt.plot(hist["loss"], label="train")
        plt.plot(hist["val_loss"], label="val")
        plt.title("Learning curve – Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.legend()

        plt.tight_layout()
        plt.savefig("learning_curve_loss.png")
        # Accuracy
        plt.figure()
        plt.plot(hist["accuracy"], label="train")
        plt.plot(hist["val_accuracy"], label="val")
        plt.title("Learning curve – Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        plt.legend()

        plt.tight_layout(),
        plt.savefig("learning_curve_accuracy.png")

if args.image:
    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    imgage = (Image
              .open(args.image)
              .convert("L")
              .resize((28, 28))) # mozliwe znieksztalcenie proporcji alternatywa ImageOps.contain
    x = np.array(imgage).astype(np.float32) / 255.0
    if x.mean() > 0.5:
        x = 1.0 - x

    probs = model.predict(x[None, ...])[0]
    pred = int(np.argmax(probs))
    print(f"Prediction: {pred} (Accuracy: {probs[pred]:.2%})")
