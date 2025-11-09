# lab3/wine.py
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_IN  = HERE / "wine.csv"
CSV_CLEAN = HERE / "wine_clean.csv"
CSV_OUT = HERE / "wine_shuffled.csv"

COLUMN_NAMES = [
    "class","alcohol","malic_acid","ash","alcalinity_of_ash","magnesium",
    "total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins",
    "color_intensity","hue","od280/od315_of_diluted_wines","proline"
]

def main(seed: int = 42):
    if not CSV_IN.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {CSV_IN}")

    df = pd.read_csv(CSV_IN, header=None, names=COLUMN_NAMES)
    print(f"[INFO] shape: {df.shape}")
    print(f"[INFO] classes: {sorted(df['class'].unique())}")

    df.to_csv(CSV_CLEAN, index=False)

    X = df.drop(columns=["class"]).to_numpy(dtype=np.float32)
    y = df["class"].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X_shuf = X[perm]
    y_shuf = y[perm]

    df_shuf = pd.DataFrame(X_shuf, columns=COLUMN_NAMES[1:])
    df_shuf.insert(0, "class", y_shuf)
    df_shuf.to_csv(CSV_OUT, index=False)

    print(f"[OK] Zapisano: {CSV_OUT.name}")
    print("[SAMPLE] y[:10] po tasowaniu:", y_shuf[:10])
    print("[CHECK] Liczebności klas:\n", df_shuf["class"].value_counts().sort_index())

if __name__ == "__main__":
    main()
