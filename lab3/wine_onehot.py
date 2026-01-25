from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_IN = HERE / "wine_shuffled.csv"
NPZ_OUT = HERE / "wine_onehot.npz"

COLUMN_NAMES = [
    "class","alcohol","malic_acid","ash","alcalinity_of_ash","magnesium",
    "total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins",
    "color_intensity","hue","od280/od315_of_diluted_wines","proline"
]

def main():
    if not CSV_IN.exists():
        raise FileNotFoundError(f"Brak {CSV_IN}. Najpierw uruchom tasowanie (wine.py).")

    df = pd.read_csv(CSV_IN)
    assert list(df.columns) == COLUMN_NAMES, "Nieoczekiwane nagłówki w CSV."

    X = df.drop(columns=["class"]).to_numpy(dtype=np.float32)
    y = df["class"].to_numpy(dtype=np.int64)

    num_classes = 3
    y0 = y - 1
    assert set(np.unique(y0)) == {0,1,2}, f"Klasy nie są 1..3: {np.unique(y)}"
    Y = np.eye(num_classes, dtype=np.float32)[y0]

    print("[INFO] X shape:", X.shape)
    print("[INFO] Y (one-hot) shape:", Y.shape)
    print("[SAMPLE] y surowe:", y[:8])
    print("[SAMPLE] Y one-hot:\n", Y[:5])

    np.savez(NPZ_OUT, X=X, y=y, Y=Y)
    print(f"[OK] Zapisano macierze do: {NPZ_OUT.name}")

if __name__ == "__main__":
    main()
