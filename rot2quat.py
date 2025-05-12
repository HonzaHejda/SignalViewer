import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

rot_cols  = [f"Mat[{r}][{c}]" for r in range(3) for c in range(3)]
quat_cols = ["Quat_q0", "Quat_q1", "Quat_q2", "Quat_q3"]

def mat_to_quat(row):
    m = row.values.astype(float)
    r00,r01,r02, r10,r11,r12, r20,r21,r22 = m
    t = r00 + r11 + r22
    if t > 0:
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
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


def process_folder(folder:Path) -> None:
    if not folder.is_dir():
        sys.exit(f"'{folder}' není existující složka!")

    # ↪  maska '?_MT_*.txt' – přesně jeden znak před '_MT_'
    for infile in folder.glob("?_MT_*.txt"):
        outfile = infile.with_name(f"{infile.stem}_quat.csv")
        try:
            df = pd.read_csv(infile.resolve(), skiprows=3, sep="\t", dtype=float, header=1)
            df[quat_cols] = df[rot_cols].apply(mat_to_quat, axis=1, result_type="expand")
            df.to_csv(outfile.resolve(), index=False)

            print(f"✔︎  {infile.name}  →  {outfile.name}")
        except Exception as exc:
            print(f"✖︎  {infile.name}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch konverze MT souborů na variantu s kvaterniony")
    ap.add_argument("folder", type=Path, help="Složka, ve které hledat soubory")
    args = ap.parse_args()
    process_folder(args.folder.expanduser())


if __name__ == "__main__":
    main()
