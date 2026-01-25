import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- dane ---
df = pd.read_csv('penguins_size.csv')
cols = ['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex']
df = df[cols].dropna()
df = df[df['sex'].isin(['MALE','FEMALE'])]

X = df[['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g']].astype('float32').values
y = (df['sex'] == 'MALE').astype('int32').values  # 1=MALE, 0=FEMALE

stratify_opt = y if pd.Series(y).value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_opt
)

def train_model(X_train, y_train, epochs=20, batch_size=32, lr=0.01):
    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(X_train)

    inputs = tf.keras.Input(shape=(4,))
    x = norm(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    return model, history

MODEL_PATH = 'penguins_sex.keras'

if os.path.isfile(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Existing model loaded: {MODEL_PATH}")
else:
    model, history = train_model(X_train, y_train, epochs=30, batch_size=32, lr=0.01)
    model.save(MODEL_PATH)
    print(f"New model saved to: {MODEL_PATH}")

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}, loss: {loss:.4f}")
