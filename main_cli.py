from __future__ import annotations

import argparse
import math
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf

def rotation_matrix_np(angle: float, *, degrees: bool = True) -> np.ndarray:
    """2x2 macierz obrotu CCW (NumPy)."""
    theta = np.deg2rad(angle) if degrees else float(angle)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)

def rotate_points_np(points: np.ndarray, angle: float, *, degrees: bool = True) -> np.ndarray:
    """Obrót punktów 2D (NumPy)."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[-1] != 2:
        raise ValueError("Oczekiwano punktów o kształcie (..., 2).")
    r = rotation_matrix_np(angle, degrees=degrees)
    return pts @ r.T

def rotation_matrix_tf(angle, *, degrees: bool = True) -> tf.Tensor:
    """2x2 macierz obrotu CCW (TensorFlow)."""
    ang = tf.convert_to_tensor(angle, dtype=tf.float32)
    theta = tf.where(
        tf.convert_to_tensor(degrees),
        ang * tf.constant(math.pi / 180.0, dtype=tf.float32),
        ang,
    )
    c, s = tf.cos(theta), tf.sin(theta)
    return tf.stack([[c, -s], [s, c]], axis=0)


@tf.function
def rotate_points_tf(points, angle, *, degrees: bool = True) -> tf.Tensor:
    """Obrót punktów 2D (TensorFlow) — działa także dla pojedynczego punktu."""
    pts = tf.convert_to_tensor(points)
    if pts.dtype != tf.float32:
        pts = tf.cast(pts, tf.float32)

    if pts.shape.rank is None or pts.shape[-1] != 2:
        raise ValueError("Oczekiwano punktów o kształcie (..., 2).")
    pts2d = tf.reshape(pts, (-1, 2))

    r = rotation_matrix_tf(angle, degrees=degrees)
    out = tf.matmul(pts2d, tf.transpose(r))
    return out


def parse_cli_points(vals: Iterable[float]) -> np.ndarray:
    """Z listy [x y x y ...] robi tablicę Nx2."""
    arr = np.asarray(list(vals), dtype=float)
    if arr.size % 2 != 0:
        raise SystemExit("Liczba współrzędnych musi być parzysta: x y x y ...")
    return arr.reshape(-1, 2)

def pretty_print_points(before: np.ndarray, after: np.ndarray) -> None:
    """Ładne wypisanie tabeli punktów przed i po obrocie."""
    print("\nOBRÓT PUNKTÓW 2D WOKÓŁ (0,0)")
    print(f"{'Punkt oryginalny':>20}  -->  {'Po obrocie':<20}")
    print("-" * 45)
    for (x, y), (xr, yr) in zip(before, after):
        print(f"({x:6.3f}, {y:6.3f})     →   ({xr:7.3f}, {yr:7.3f})")
    print("-" * 45)

def _ensure_column(b: tf.Tensor) -> Tuple[tf.Tensor, bool]:
    """Zwraca (b_kolumnowo, czy_wejście_było_wektorem)."""
    was_vector = b.shape.rank is None or b.shape.rank == 1
    if was_vector:
        b = tf.expand_dims(b, axis=-1)
    return b, was_vector

def solve_system_auto(A, b) -> np.ndarray:
    """Najpierw `solve`, w razie problemu — `lstsq` (TensorFlow)."""
    A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
    b_tf = tf.convert_to_tensor(b, dtype=tf.float32)
    b_tf, was_vector = _ensure_column(b_tf)

    try:
        x = tf.linalg.solve(A_tf, b_tf)
    except tf.errors.InvalidArgumentError:
        x = tf.linalg.lstsq(A_tf, b_tf, fast=False)

    if was_vector:
        x = tf.squeeze(x, axis=-1)
    return x.numpy()

def solve_system_solve(A, b) -> np.ndarray:
    """Wymuszone `solve` — błąd jeśli macierz osobliwa."""
    A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
    b_tf = tf.convert_to_tensor(b, dtype=tf.float32)
    b_tf, was_vector = _ensure_column(b_tf)
    x = tf.linalg.solve(A_tf, b_tf)
    if was_vector:
        x = tf.squeeze(x, axis=-1)
    return x.numpy()

def solve_system_lstsq(A, b) -> np.ndarray:
    """Least squares (dla układów nad/nieokreślonych)."""
    A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
    b_tf = tf.convert_to_tensor(b, dtype=tf.float32)
    b_tf, was_vector = _ensure_column(b_tf)
    x = tf.linalg.lstsq(A_tf, b_tf, l2_regularizer=0.0, fast=False)
    if was_vector:
        x = tf.squeeze(x, axis=-1)
    return x.numpy()

def parse_matrix_and_vector(
    A_vals: Iterable[float] | None,
    b_vals: Iterable[float] | None,
    n: int | None,
    packed: Iterable[float] | None,
    *,
    m: int | None = None,
    k: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parsuje wejście macierzy/wektora.
    Wspiera:
      - --packed (tylko kwadratowe: n^2 + n),
      - --A/--b z wymiarami --m --k (prostokątne).
    """
    if packed is not None:
        nums = np.asarray(list(packed), dtype=float)
        total = nums.size
        found = False
        root = int(np.sqrt(total)) + 2
        for cand in range(1, root + 1):
            if cand * cand + cand == total:
                n = cand
                found = True
                break
        if not found:
            raise SystemExit(f"Liczba wartości ({total}) nie pasuje do n² + n.")
        A = nums[: n * n].reshape(n, n)
        b = nums[n * n :]
        return A, b

    if A_vals is None or b_vals is None or m is None or k is None:
        raise SystemExit(
            "Podaj --packed LUB ( --A --b --m --k ). "
            "Dla prostokątnych A użyj --A/--b oraz wymiarów --m --k."
        )

    A_list = list(A_vals)
    b_list = list(b_vals)
    if len(A_list) != m * k:
        raise SystemExit(f"--A musi mieć {m*k} elementów (podano {len(A_list)}).")
    if len(b_list) != m:
        raise SystemExit(f"--b musi mieć {m} elementów (podano {len(b_list)}).")

    A = np.array(A_list, dtype=float).reshape(m, k)
    b = np.array(b_list, dtype=float)
    return A, b

def pretty_print_vector(x: np.ndarray, title: str = "WYNIK", ndigits: int = 6) -> None:
    """Ładne wypisanie rozwiązań."""
    x_round = np.round(np.asarray(x, dtype=float), ndigits)
    print(f"\n=== {title} ===")
    if x_round.ndim == 0:
        print(f"x1 = {float(x_round):.{ndigits}f}")
        return
    for i, val in enumerate(x_round, start=1):
        if abs(val) < 1e-8:
            val = 0.0
        print(f"x{i} = {val:.{ndigits}f}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zadanie 5: CLI do obrotu punktów i rozwiązywania układów równań (NumPy/TensorFlow).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Przykłady:\n"
            "  Obrót (NumPy):\n"
            "    python main_cli.py rotate --backend np --angle 90 --points 1 0  0 1  1 1 --pretty\n"
            "  Obrót (TensorFlow, radiany):\n"
            "    python main_cli.py rotate --backend tf --angle 1.57079632679 --radians --points 1 0\n"
            "  Układ (pakiet n²+n):\n"
            "    python main_cli.py solve --method auto --packed 3 2 1 2  5 5\n"
            "  Least squares (prostokątne A m×k):\n"
            "    python main_cli.py solve --method lstsq --A 1 0  0 1  1 1 --b 1 1 2 --m 3 --k 2\n"
        ),
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rot = sub.add_parser("rotate", help="Obrót punktów 2D wokół (0,0).")
    p_rot.add_argument("--backend", choices=("np", "tf"), default="np", help="Silnik: NumPy lub TensorFlow.")
    p_rot.add_argument("--angle", type=float, required=True, help="Kąt obrotu (domyślnie w stopniach).")
    p_rot.add_argument("--radians", action="store_true", help="Jeśli podano — `angle` w radianach.")
    p_rot.add_argument(
        "--points",
        type=float,
        nargs="+",
        required=True,
        help="Lista współrzędnych: x y x y ... (parzysta liczba wartości).",
    )
    p_rot.add_argument("--pretty", action="store_true", help="Ładna tabela wyników.")

    p_sol = sub.add_parser("solve", help="Rozwiązanie układu A·x = b (TensorFlow).")
    p_sol.add_argument(
        "--method",
        choices=("auto", "solve", "lstsq"),
        default="auto",
        help="auto: próbuj solve, w razie potrzeby lstsq; 'solve' lub 'lstsq' wymusza metodę.",
    )

    p_sol.add_argument(
        "--packed",
        type=float,
        nargs="+",
        help="Jedna lista (n^2 + n): najpierw A (wierszami, kwadratowa), potem b.",
    )
    p_sol.add_argument("--A", type=float, nargs="+", help="Współczynniki A (m*k liczb, wierszami).")
    p_sol.add_argument("--b", type=float, nargs="+", help="Współczynniki b (m liczb).")
    p_sol.add_argument("--m", type=int, help="Liczba wierszy A (i długość b).")
    p_sol.add_argument("--k", type=int, help="Liczba kolumn A.")

    p_sol.add_argument("--title", type=str, default="WYNIK", help="Nagłówek wydruku.")
    p_sol.add_argument("--ndigits", type=int, default=6, help="Zaokrąglenie wyników.")

    return parser

def cmd_rotate(args: argparse.Namespace) -> None:
    pts = parse_cli_points(args.points)
    deg = not args.radians

    if args.backend == "np":
        out = rotate_points_np(pts, args.angle, degrees=deg)
    else:
        out_tf = rotate_points_tf(pts, args.angle, degrees=deg)
        out = out_tf.numpy()

    res = np.round(out, 6)
    if args.pretty:
        pretty_print_points(pts, res)
    else:
        np.set_printoptions(precision=6, suppress=True)
        print(res)

def cmd_solve(args: argparse.Namespace) -> None:
    A, b = parse_matrix_and_vector(args.A, args.b, None, args.packed, m=args.m, k=args.k)

    A_np = np.asarray(A, dtype=float)
    if args.method == "solve" and A_np.shape[0] == A_np.shape[1]:
        det = float(np.linalg.det(A_np))
        if abs(det) < 1e-10:
            raise SystemExit("Macierz osobliwa — układ nie ma jednoznacznego rozwiązania (det≈0).")

    try:
        if args.method == "solve":
            x = solve_system_solve(A, b)
        elif args.method == "lstsq":
            x = solve_system_lstsq(A, b)
        else:
            x = solve_system_auto(A, b)
    except tf.errors.InvalidArgumentError as e:
        raise SystemExit(f"Błąd rozwiązywania (solve): {e}\nSugeruję użyć --method lstsq.")

    b_np = np.asarray(b, dtype=float)
    x_np = np.atleast_1d(x)
    res = A_np @ x_np - b_np
    res_norm = float(np.linalg.norm(res))

    eps_res = 1e-9
    if res_norm <= eps_res:
        print(f"(dokładne rozwiązanie, ||A x - b|| = {res_norm:.3e})")
    else:
        print(f"(rozwiązanie przybliżone, ||A x - b|| = {res_norm:.3e})")

    pretty_print_vector(x, title=args.title, ndigits=args.ndigits)

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "rotate":
        cmd_rotate(args)
    elif args.cmd == "solve":
        cmd_solve(args)
    else:
        parser.error("Nieznana komenda.")


if __name__ == "__main__":
    main()
