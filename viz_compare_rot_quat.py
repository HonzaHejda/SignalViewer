from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – nutné pro 3‑D projekci

# ---------------------------------------------------------------------------
# Matematika
# ---------------------------------------------------------------------------

def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Robustní převod 3×3 rotační matice → (normalizovaný) kvaternion (w, x, y, z)."""
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = R.flatten()
    t = r00 + r11 + r22
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        qw = 0.25 / s
        qx = (r21 - r12) * s
        qy = (r02 - r20) * s
        qz = (r10 - r01) * s
    elif r00 > r11 and r00 > r22:
        s = 2.0 * np.sqrt(1.0 + r00 - r11 - r22)
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * np.sqrt(1.0 + r11 - r00 - r22)
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r22 - r00 - r11)
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / np.linalg.norm(q)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Kvaternion (w, x, y, z) → 3×3 rotační matice."""
    qw, qx, qy, qz = q
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz
    qwqx, qwqy, qwqz = qw * qx, qw * qy, qw * qz
    qxqy, qxqz, qyqz = qx * qy, qx * qz, qy * qz
    return np.array([
        [1 - 2 * (qy2 + qz2), 2 * (qxqy - qwqz), 2 * (qxqz + qwqy)],
        [2 * (qxqy + qwqz), 1 - 2 * (qx2 + qz2), 2 * (qyqz - qwqx)],
        [2 * (qxqz - qwqy), 2 * (qyqz + qwqx), 1 - 2 * (qx2 + qy2)],
    ])

# ---------------------------------------------------------------------------
# CSV utility
# ---------------------------------------------------------------------------

def extract_matrix(df: pd.DataFrame, row: int) -> np.ndarray:
    rot_cols: List[str] = [f"Mat[{r}][{c}]" for r in range(3) for c in range(3)]
    return df.loc[row, rot_cols].to_numpy(float).reshape(3, 3)


def extract_quaternion(df: pd.DataFrame, row: int) -> np.ndarray | None:
    quat_cols = ["Quat_q0", "Quat_q1", "Quat_q2", "Quat_q3"]
    if all(c in df.columns for c in quat_cols):
        return df.loc[row, quat_cols].to_numpy(float)
    return None

# ---------------------------------------------------------------------------
# Vizualizace
# ---------------------------------------------------------------------------

def draw_axes(ax: Axes3D, R: np.ndarray, title: str):
    ax.cla()  # vyčistit předchozí
    for i in range(3):
        ax.quiver(0, 0, 0, R[0, i], R[1, i], R[2, i], linewidth=2)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=25, azim=135)

# ---------------------------------------------------------------------------
# Hlavní smyčka
# ---------------------------------------------------------------------------

def animate(df: pd.DataFrame, start: int, interval: float, skip: int):
    stop = {"quit": False}

    def on_key(_event):
        stop["quit"] = True

    plt.ion()
    fig = plt.figure("Xsens vizualizace", figsize=(10, 5))
    ax_mat = fig.add_subplot(1, 2, 1, projection="3d")
    ax_quat = fig.add_subplot(1, 2, 2, projection="3d")
    fig.canvas.mpl_connect("key_press_event", on_key)

    total = len(df)

    for i in range(start, total, skip):
        if stop["quit"]:
            break
        # --- rotační matice ---
        R_mat = extract_matrix(df, i)
        draw_axes(ax_mat, R_mat, f"Mat[{i}]")

        # --- kvaternion (pokud není v CSV, dopočti) ---
        q = extract_quaternion(df, i)
        if q is None:
            q = mat_to_quat(R_mat)
        R_quat = quat_to_mat(q / np.linalg.norm(q))
        draw_axes(ax_quat, R_quat, f"Quat[{i}]")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(interval)

    plt.ioff()
    plt.show()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interaktivní vizualizace Xsens orientace")
    parser.add_argument("csv", type=Path, help="Cesta k CSV exportu")
    parser.add_argument("--interval", type=float, default=0.5, help="Interval snímků v sekundách")
    parser.add_argument("--start", type=int, default=0, help="Počáteční index řádku")
    parser.add_argument("--skip", type=int, default=100, help="Kolik položek přeskočit při iteraci")
    args = parser.parse_args()

    if not args.csv.is_file():
        sys.exit(f"Soubor '{args.csv}' neexistuje!")

    df = pd.read_csv(args.csv)
    animate(df, args.start, args.interval, args. skip)


if __name__ == "__main__":
    main()
