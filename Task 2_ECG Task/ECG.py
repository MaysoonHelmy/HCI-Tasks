import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.fft import dct
import pywt
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def load_signal(file_path):
    data = np.loadtxt(file_path)
    return data[:, 1] if (data.ndim == 2 and data.shape[1] >= 2) else data


def preprocess_ecg(raw_data, fs=250):
    nyquist = 0.5 * fs
    low, high = 1 / nyquist, 40 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    return signal.filtfilt(b, a, raw_data)


def pan_tompkins(sig, fs=250):
    diff = np.convolve(sig, [1, 2, 0, -2, -1], mode='same')
    squared = diff ** 2
    win = int(0.03 * fs)
    mwa = np.convolve(squared, np.ones(win) / win, mode='same')
    threshold = np.mean(mwa)
    peaks, _ = signal.find_peaks(mwa, distance=int(0.6 * fs), height=threshold)
    return peaks


def detect_qrs(sig, r_peaks, fs=250):
    qrs_points = []
    q_win = int(0.04 * fs)
    s_win = int(0.04 * fs)

    for r in r_peaks:
        q_start = max(0, r - q_win)
        q = q_start + np.argmin(sig[q_start:r]) if r > q_start else r

        s_end = min(len(sig), r + s_win)
        s = r + np.argmin(sig[r:s_end]) if s_end > r else r

        qrs_points.append((q, r, s))

    return qrs_points


def ac_dct_features(segment):
    ac = np.correlate(segment, segment, mode='full')
    ac = ac[len(ac)//2:]
    ac = ac[:100]
    dct_coeff = dct(ac, type=2, norm='ortho')
    return dct_coeff[:20]


def extract_features(sig, qrs_points):
    if len(qrs_points) == 0:
        return np.zeros(25)

    q, r, s = qrs_points[len(qrs_points)//2]

    qr = r - q
    rs = s - r
    slope = (sig[s] - sig[q]) / (s - q + 1e-6)

    start = max(0, r - 75)
    end = min(len(sig), r + 75)
    segment = sig[start:end]

    coeffs = pywt.wavedec(segment, 'db4', level=2)
    dwt_a2 = np.mean(coeffs[0])
    dwt_d2 = np.mean(coeffs[1])

    acdct = ac_dct_features(segment)

    return np.array([qr, rs, slope, dwt_a2, dwt_d2] + list(acdct))


def feature_names():
    return [
        "QR_interval",
        "RS_interval",
        "Slope",
        "DWT_A2_mean",
        "DWT_D2_mean"
    ] + [f"AC_DCT_{i+1}" for i in range(20)]


def save_to_excel(features_dict):
    names = feature_names()
    df = pd.DataFrame.from_dict(features_dict, orient='index', columns=names)
    df.to_excel("ECG_Features.xlsx")


def compute_distinct_feature(ali, moh):
    mean1 = np.mean(ali, axis=0)
    mean2 = np.mean(moh, axis=0)
    std1 = np.std(ali, axis=0)
    std2 = np.std(moh, axis=0)
    score = np.abs(mean1 - mean2) / (std1 + std2 + 1e-6)
    return np.argmax(score)


class ECGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Biometric Identification")
        self.root.geometry("1000x800")

        tk.Label(root, text="ECG Analysis System", font=('Helvetica', 18, 'bold')).pack(pady=10)

        self.btn = tk.Button(root, text="START CLASSIFICATION", command=self.run_analysis)
        self.btn.pack(pady=10)

        self.lbl_match = tk.Label(root, text="Match Result: --", font=('Arial', 14))
        self.lbl_match.pack()

        self.lbl_feat = tk.Label(root, text="Best Feature: --", font=('Arial', 12))
        self.lbl_feat.pack()

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_analysis(self):
        try:
            data_map = {
                "Ali": "ECG_Ali.txt",
                "Mohamed": "ECG_Mohamed.txt",
                "Test": "Test signal.txt"
            }

            feats = {}
            signals = {}

            for name, path in data_map.items():
                raw = load_signal(path)
                clean = preprocess_ecg(raw)
                r_peaks = pan_tompkins(clean)
                qrs = detect_qrs(clean, r_peaks)
                feats[name] = extract_features(clean, qrs)
                signals[name] = clean

            ali = feats["Ali"]
            moh = feats["Mohamed"]
            test = feats["Test"]

            ali_mat = np.array([ali])
            moh_mat = np.array([moh])

            best_idx = compute_distinct_feature(ali_mat, moh_mat)

            self.lbl_feat.config(text=f"Most Distinctive Feature: {feature_names()[best_idx]}")

            d1 = np.linalg.norm(test - ali, ord=2)
            d2 = np.linalg.norm(test - moh, ord=2)

            winner = "Ali" if d1 < d2 else "Mohamed"
            self.lbl_match.config(text=f"IDENTIFIED AS: {winner}")

            self.ax1.clear()
            self.ax2.clear()

            self.ax1.plot(signals["Test"][500:1500])
            self.ax1.set_title("Test Signal")

            self.ax2.plot(signals[winner][500:1500])
            self.ax2.set_title("Matched Signal")

            self.canvas.draw()

            save_to_excel({
                "Ali": ali,
                "Mohamed": moh,
                "Test": test
            })

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.mainloop()
    plt.close('all')