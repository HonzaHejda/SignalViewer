import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from PyQt6.QtWidgets import QApplication, QFileDialog
import plotly.io as pio
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

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


def generate_plot_titles(num_emg_files, num_imu_files):
    signal_list = []

    for i in range(1, num_emg_files + 1):
        signal_list.append(f"Signal {i} - EMG")
        if num_imu_files > 0:
            signal_list.append(f"Signal {i} - Acceleration")
            signal_list.append(f"Signal {i} - Orientation")

    return signal_list


# Načtení datových souborů
emg_file_paths = load_csv_files()
if not emg_file_paths:
    print("No files selected. Exiting.")
    exit()

imu_file_paths = load_csv_files()
if len(imu_file_paths) != 0 and len(imu_file_paths) != len(emg_file_paths):
    print("The number of IMU files and the number of IMU files is not equal. Exiting.")
    exit()

num_emg_files = len(emg_file_paths)
num_imu_files = len(imu_file_paths)

# Vytvoření figure se subgrafy
fig = sp.make_subplots(rows=num_emg_files+2*num_imu_files, cols=1,
                       shared_xaxes=True, shared_yaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=generate_plot_titles(num_emg_files, num_imu_files))

# Načtení a zpracování každého souboru
for i, file_path in enumerate(emg_file_paths):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["Sample", "Signal"], dtype={"Sample": int, "Signal": float})
    df["Time"] = df["Sample"] / 200  # Přepočet čísla vzorku na čas

    # Výpočet obálky signálu pomocí neurokit2
    cleaned = custom_highpass(df["Signal"], cutoff=10, fs=200)
    amplitude = nk.emg_amplitude(cleaned)

    if num_imu_files > 0:
        row = i * 3

    fig.add_trace(go.Scatter(x=df["Time"], y=amplitude, mode='lines',
                             name=f"Signal {i+1}"),
                  row=(i*3 if num_imu_files>0 else i) + 1, col=1)

    # Zpracování IMU souboru
    if num_imu_files > 0:
        imu_df = pd.read_csv(imu_file_paths[i], sep="\t", comment='/')
        imu_df["Time"] = imu_df.index / 100  # Vzorkovací frekvence 100 Hz

        # Přidání akcelerací
        fig.add_trace(
            go.Scatter(x=imu_df["Time"], y=imu_df["Acc_X"], mode='lines', name=f"Acc_X {i+1} - IMU"),
            row=(i*3 if num_imu_files>0 else i) + 2, col=1)
        fig.add_trace(
            go.Scatter(x=imu_df["Time"], y=imu_df["Acc_Y"], mode='lines', name=f"Acc_Y {i+1} - IMU"),
            row=(i*3 if num_imu_files>0 else i) + 2, col=1)
        fig.add_trace(
            go.Scatter(x=imu_df["Time"], y=imu_df["Acc_Z"], mode='lines', name=f"Acc_Z {i+1} - IMU"),
            row=(i*3 if num_imu_files>0 else i) + 2, col=1)


        # Výpočet Eulerových úhlů
        rotation_matrices = imu_df.iloc[:, 9:18].to_numpy().reshape(-1, 3, 3)
        eulers = R.from_matrix(rotation_matrices).as_euler('xyz', degrees=True)
        roll, pitch, yaw = eulers[:, 0], eulers[:, 1], eulers[:, 2]

        fig.add_trace(go.Scatter(x=imu_df["Time"], y=roll, mode='lines', name=f"Roll {i + 1} - IMU"),
                      row=(i*3 if num_imu_files>0 else i) + 3, col=1)
        fig.add_trace(go.Scatter(x=imu_df["Time"], y=pitch, mode='lines', name=f"Pitch {i + 1} - IMU"),
                      row=(i*3 if num_imu_files>0 else i) + 3, col=1)
        fig.add_trace(go.Scatter(x=imu_df["Time"], y=yaw, mode='lines', name=f"Yaw {i + 1} - IMU"),
                      row=(i*3 if num_imu_files>0 else i) + 3, col=1)

# Nastavení layoutu
fig.update_layout(
    title="EMG signals",
    xaxis_title="Time (s)",
    height=600 * num_emg_files,
    showlegend=False,
    template="plotly_white"
)

# Nastavení zobrazení grafu v novém okně
pio.renderers.default = "browser"
fig.show()