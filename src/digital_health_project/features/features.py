import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class HandFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extractor for bilateral accelerometer signals (active vs mirror hand).

    The transformer computes:
    
    1. Univariate features per signal (active & mirror):
       - mean
       - std
       - median
       - max
       - 90th percentile (q90)
       - sum
       - fraction of values > 0

    2. Bilateral features:
       - mean difference
       - mean absolute difference
       - sum ratio
       - asymmetry index
       - zero-lag correlation
       - max cross-correlation and lag (within ±max_lag)

    3. Multiple feature selection modes:
       - active_only: only active-side features
       - active_mirror: active + mirror features (no bilateral)
       - bilateral_only: only interaction features
       - all_features: everything

    Parameters
    ----------
    mode : str, default="all_features"
        Feature subset to use:
        - "active_only"
        - "active_mirror"
        - "bilateral_only"
        - "all_features"

    max_lag : int, default=3
        Maximum lag (in samples) for cross-correlation search.
    """

    def __init__(self, mode="all_features", max_lag=3):
        self.mode = mode
        self.max_lag = max_lag

    def fit(self, X, y=None):
        return self

    # ---------- UNIVARIATE FEATURES ----------
    def summarize_signal(self, signal):
        signal = np.asarray(signal)

        return {
            "mean": np.mean(signal),
            "std": np.std(signal),
            "median": np.median(signal),
            "max": np.max(signal),
            "q90": np.percentile(signal, 90),
            "sum": np.sum(signal),
            "nonzero_frac": np.mean(signal > 0)
        }

    # ---------- CROSS-CORRELATION ----------
    def max_abs_xcorr(self, x, y):
        best_corr = 0
        best_lag = 0

        for lag in range(-self.max_lag, self.max_lag + 1):

            if lag < 0:
                corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
            else:
                corr = np.corrcoef(x, y)[0, 1]

            if np.isnan(corr):
                corr = 0

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return best_corr, best_lag

    # ---------- TRANSFORM ----------
    def transform(self, X):
        all_features = []

        for window in X:
            hand_A = window[:, :3]
            hand_N = window[:, 3:]

            vm_A = np.linalg.norm(hand_A, axis=1)
            vm_N = np.linalg.norm(hand_N, axis=1)

            total_A = np.sum(hand_A, axis=1)
            total_N = np.sum(hand_N, axis=1)

            signals = {
                "Axis1": (hand_A[:, 0], hand_N[:, 0]),
                "Axis2": (hand_A[:, 1], hand_N[:, 1]),
                "Axis3": (hand_A[:, 2], hand_N[:, 2]),
                "VM": (vm_A, vm_N),
                "Total": (total_A, total_N),
            }

            feat = {}
            eps = 1e-8

            # ---------- UNIVARIATE ----------
            for name, (a, m) in signals.items():
                for k, v in self.summarize_signal(a).items():
                    feat[f"{name}_active_{k}"] = v
                for k, v in self.summarize_signal(m).items():
                    feat[f"{name}_mirror_{k}"] = v

            # ---------- BILATERAL ----------
            for name, (a, m) in signals.items():
                diff = a - m

                feat[f"{name}_bilateral_mean_diff"] = np.mean(diff)
                feat[f"{name}_bilateral_abs_mean_diff"] = np.mean(np.abs(diff))

                feat[f"{name}_bilateral_sum_ratio"] = np.sum(m) / (np.sum(a) + eps)

                feat[f"{name}_bilateral_asymmetry"] = (
                    (np.sum(a) - np.sum(m)) /
                    (np.sum(a) + np.sum(m) + eps)
                )

                corr = np.corrcoef(a, m)[0, 1]
                feat[f"{name}_bilateral_corr0"] = 0 if np.isnan(corr) else corr

                max_corr, lag = self.max_abs_xcorr(a, m)
                feat[f"{name}_bilateral_maxcorr_abs"] = abs(max_corr)
                feat[f"{name}_bilateral_lag_at_maxcorr"] = lag

            # ---------- MODE FILTERING ----------
            if self.mode == "active_only":
                feat = {k: v for k, v in feat.items() if "_active_" in k}

            elif self.mode == "active_mirror":
                feat = {k: v for k, v in feat.items()
                        if "_active_" in k or "_mirror_" in k}

            elif self.mode == "bilateral_only":
                feat = {k: v for k, v in feat.items() if "_bilateral_" in k}

            elif self.mode == "all_features":
                pass

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            all_features.append(list(feat.values()))

        return np.array(all_features)