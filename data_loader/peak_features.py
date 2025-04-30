from scipy.signal import find_peaks
import numpy as np

def extract_peak_features(window_df, sampling_rate):
    features = []
    time_diffs = []

    for col in ['g_x', 'g_y', 'g_z', 'ac_x', 'ac_y', 'ac_z']:
        signal = window_df[col].values
        peaks, _ = find_peaks(signal)

        # Konwertuje indeksy peakow na timestamp
        if len(peaks) > 1:
            peak_times = peaks / sampling_rate
            time_diff = np.diff(peak_times)
            avg_time_between_peaks = np.mean(time_diff)
            std_time_between_peaks = np.std(time_diff)
            features.append(avg_time_between_peaks)
            features.append(std_time_between_peaks)
        else:
            # Nie wystarczajaca liczba peakow
            features.extend([0, 0])

        features.append(len(peaks))

    return features
