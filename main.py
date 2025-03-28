import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import tkinter as tk
from PyQt6.QtWidgets import QApplication, QFileDialog
import plotly.io as pio
import neurokit2 as nk
from scipy.signal import butter, filtfilt

# Funkce pro načtení více souborů
def load_csv_files():
    app = QApplication([])
    file_paths, _ = QFileDialog.getOpenFileNames(
        None, "Select Files", "", "CSV/TXT Files (*.csv *.txt)"
    )
    return file_paths


def custom_highpass(signal, cutoff=10, fs=200, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered


# Načtení datových souborů
file_paths = load_csv_files()
if not file_paths:
    print("No files selected. Exiting.")
    exit()

num_files = len(file_paths)

# Vytvoření figure se subgrafy
fig = sp.make_subplots(rows=num_files, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=[f"Signal {i+1}" for i in range(num_files)])

# Načtení a zpracování každého souboru
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["Sample", "Signal"], dtype={"Sample": int, "Signal": float})
    df["Time"] = df["Sample"] / 200  # Přepočet čísla vzorku na čas

    # Výpočet obálky signálu pomocí neurokit2
    cleaned = custom_highpass(df["Signal"], cutoff=10, fs=200)
    amplitude = nk.emg_amplitude(cleaned)

    fig.add_trace(go.Scatter(x=df["Time"], y=amplitude, mode='lines', name=f"Signal {i+1}"), row=i+1, col=1)

# Nastavení layoutu
fig.update_layout(
    title="Interactive Time Series Visualization",
    xaxis_title="Time (s)",
    height=300 * num_files,
    showlegend=False,
    template="plotly_white"
)

# Nastavení zobrazení grafu v novém okně
pio.renderers.default = "browser"
fig.show()