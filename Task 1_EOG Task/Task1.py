import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial.distance import euclidean
from scipy.linalg import solve

class EOGAnalyzer:
    def __init__(self):
        self.sampling_rate = 176
        self.signals = None
        self.signal_names = None
        self.processed_signals = None
        self.features_df = None
        self.test_signal_raw = None
        self.matching_signal_raw = None
        
    def load_horizontal_signals(self, filepath):
        df = pd.read_excel(filepath, header=None)
        self.signal_names = df.iloc[-1, :].values.astype(str)
        self.signals = df.iloc[:-1, :].values.astype(float).T
        return self.signals, self.signal_names
    
    def load_test_signal(self, filepath):
        self.test_signal_raw = np.loadtxt(filepath, dtype=float)
        return self.test_signal_raw
    
    def preprocess_signal(self, signal_data):
        signal_data = np.array(signal_data, dtype=float)
        nyquist = 0.5 * self.sampling_rate
        low = 0.5 / nyquist
        high = 30.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        detrended = filtered - np.mean(filtered)
        return detrended
    
    def preprocess_all_signals(self):
        self.processed_signals = np.array([self.preprocess_signal(sig) for sig in self.signals])
        return self.processed_signals
    
    def calculate_ar_coefficients(self, signal_data, order=3):
        n = len(signal_data)
        r = np.zeros(order + 1)
        
        for k in range(order + 1):
            for i in range(k, n):
                r[k] += signal_data[i] * signal_data[i - k]
            r[k] /= (n - k)
        
        R = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = r[abs(i - j)]
        
        rhs = r[1:order+1]
        
        ar_coeffs = solve(R, rhs)

        
        return ar_coeffs
    
    def extract_features(self, signal_data):
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        max_peak = np.max(np.abs(signal_data))
        auc = np.trapz(np.abs(signal_data))
        ar_coeffs = self.calculate_ar_coefficients(signal_data, order=3)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'max_peak': max_peak,
            'auc': auc,
            'ar1': ar_coeffs[0],
            'ar2': ar_coeffs[1],
            'ar3': ar_coeffs[2]
        }
    
    def create_features_matrix(self):
        features_list = []
        for sig in self.processed_signals:
            features_list.append(self.extract_features(sig))
        
        self.features_df = pd.DataFrame(features_list)
        self.features_df.index = self.signal_names
        self.features_df.index.name = 'Signal_Name'
        return self.features_df
    
    def save_features_matrix(self, filename="features_matrix.xlsx"):
        self.features_df.to_excel(filename, index=True)
    
    def classify_test_signal(self, test_signal):
        processed_test = self.preprocess_signal(test_signal)
        test_features = self.extract_features(processed_test)
        test_vector = np.array(list(test_features.values()))
        
        distances = []
        for idx, sig in enumerate(self.processed_signals):
            sig_features = self.extract_features(sig)
            sig_vector = np.array(list(sig_features.values()))
            dist = euclidean(test_vector, sig_vector)
            distances.append((self.signal_names[idx], idx, dist))
        
        distances.sort(key=lambda x: x[2])
        self.matching_signal_raw = self.signals[distances[0][1]]
        
        return {
            'test_features': test_features,
            'distances': distances,
            'closest_signal': distances[0][0],
            'closest_distance': distances[0][2]
        }
    
    def run_complete_analysis(self, horizontal_file, test_file):
        self.load_horizontal_signals(horizontal_file)
        self.preprocess_all_signals()
        self.create_features_matrix()
        self.save_features_matrix()
        test_signal = self.load_test_signal(test_file)
        return self.classify_test_signal(test_signal)