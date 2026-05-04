from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.signal import spectrogram


class STFTTransformer(BaseEstimator, TransformerMixin):
    """
    STFT transformer for CNN input.

    Input:
        (samples, time, features)

    Output:
        (samples, channels, freq, time)
    """

    def __init__(
        self,
        fs=80,
        nperseg=240,        # IMPORTANT: smaller than window
        noverlap=120,
        log_transform=True,
        max_freq_bins=None
    ):
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.log_transform = log_transform
        self.max_freq_bins = max_freq_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed = []

        for window in X:
            channels = []

            for ch in range(window.shape[1]):
                f, t, Sxx = spectrogram(
                    window[:, ch],
                    fs=self.fs,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    scaling="density"
                )

                if self.log_transform:
                    Sxx = 10 * np.log10(Sxx + 1e-12)

                # Optional: keep only low frequencies
                if self.max_freq_bins is not None:
                    Sxx = Sxx[:self.max_freq_bins, :]

                channels.append(Sxx)  # (freq, time)

            # Stack → (channels, freq, time)
            spec = np.stack(channels, axis=0)

            transformed.append(spec)

        return np.array(transformed)