import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin

class HandFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features from raw accelerometer axes and vector magnitudes.
    
    Parameters:
    -----------
    mode : str, default='all'
        'all': Features for X, Y, Z for both hands, plus magnitudes and interactions.
        'subset': Only magnitudes and inter-hand interaction features.
    """
    def __init__(self, mode='all'):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def _compute_stats(self, signal):
        """Helper for standard time-domain features"""
        return [
            np.mean(signal),
            np.std(signal),
            np.max(signal) - np.min(signal), # Range
            np.percentile(signal, 75) - np.percentile(signal, 25), # IQR
            skew(signal),
            kurtosis(signal)
        ]

    def transform(self, X):
        all_features = []
        
        for window in X:
            # Split raw signals: [AccX_A, AccY_A, AccZ_A, AccX_N, AccY_N, AccZ_N]
            hand_A = window[:, :3]
            hand_N = window[:, 3:]
            
            # 1. Compute Magnitudes
            mag_A = np.sqrt(np.sum(hand_A**2, axis=1))
            mag_N = np.sqrt(np.sum(hand_N**2, axis=1))
            
            window_features = []

            # 2. Add Individual Axis Features (if 'all')
            if self.mode == 'all':
                for i in range(3): # X, Y, Z
                    window_features.extend(self._compute_stats(hand_A[:, i]))
                    window_features.extend(self._compute_stats(hand_N[:, i]))

            # 3. Add Magnitude Features
            window_features.extend(self._compute_stats(mag_A))
            window_features.extend(self._compute_stats(mag_N))

            # 4. Inter-Hand Interaction Features
            # Correlation
            corr = np.corrcoef(mag_A, mag_N)[0, 1]
            window_features.append(corr if not np.isnan(corr) else 0)
            
            # Asymmetry Ratio (Active / Non-Active)
            ratio = np.mean(mag_A) / (np.mean(mag_N) + 1e-6)
            window_features.append(ratio)
            
            # Absolute Difference
            window_features.append(np.abs(np.mean(mag_A) - np.mean(mag_N)))

            all_features.append(window_features)

        return np.array(all_features)