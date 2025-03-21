import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import tkinter as tk
from tkinter import filedialog
import plotly.io as pio

# Funkce pro načtení více souborů
def load_csv_files():
    root = tk.Tk()
    root.withdraw()  # Skryje hlavní okno Tkinter
    file_paths = filedialog.askopenfilenames(filetypes=[("Data files", "*.csv;*.txt"), ("CSV files", "*.csv"), ("Text files", "*.txt")])
    return file_paths

if sys.platform == "darwin":  # macOS
    import ctypes

    try:
        appkit = ctypes.CDLL("/System/Library/Frameworks/AppKit.framework/AppKit")
        appkit.NSApplicationLoad()
    except OSError:
        pass  # Pokud není dostupné, ignoruj

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
    fig.add_trace(go.Scatter(x=df["Time"], y=df["Signal"], mode='lines', name=f"Signal {i+1}"), row=i+1, col=1)

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