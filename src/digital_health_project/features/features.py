import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HandFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Robust feature extractor for bilateral accelerometer signals.

    Safe against:
    - NaNs
    - constant signals
    - short windows
    - divide-by-zero
    - correlation warnings
    """

    def __init__(self, mode="all_features", max_lag=3):
        self.mode = mode
        self.max_lag = max_lag

    def fit(self, X, y=None):
        return self

    # ---------- SAFE HELPERS ----------
    def safe_corr(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        if len(x) < 2 or len(y) < 2:
            return 0.0

        sx = np.std(x)
        sy = np.std(y)

        if sx < 1e-12 or sy < 1e-12:
            return 0.0

        return float(np.corrcoef(x, y)[0, 1])

    def safe_div(self, num, denom, eps=1e-8):
        if abs(denom) < eps:
            return 0.0
        return num / denom

    # ---------- UNIVARIATE FEATURES ----------
    def summarize_signal(self, signal):
        signal = np.asarray(signal)

        if signal.size == 0:
            return {
                "mean": 0.0, "std": 0.0, "median": 0.0,
                "max": 0.0, "q90": 0.0, "sum": 0.0,
                "nonzero_frac": 0.0
            }

        return {
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "median": float(np.median(signal)),
            "max": float(np.max(signal)),
            "q90": float(np.percentile(signal, 90)),
            "sum": float(np.sum(signal)),
            "nonzero_frac": float(np.mean(signal != 0))  # fixed
        }

    # ---------- CROSS-CORRELATION ----------
    def max_abs_xcorr(self, x, y):
        best_corr = 0.0
        best_lag = 0

        for lag in range(-self.max_lag, self.max_lag + 1):

            if lag < 0:
                x_slice = x[:lag]
                y_slice = y[-lag:]
            elif lag > 0:
                x_slice = x[lag:]
                y_slice = y[:-lag]
            else:
                x_slice = x
                y_slice = y

            if len(x_slice) < 2:
                corr = 0.0
            else:
                corr = self.safe_corr(x_slice, y_slice)

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return best_corr, best_lag

    # ---------- TRANSFORM ----------
    def transform(self, X):
        all_features = []

        for window in X:
            window = np.asarray(window)

            # --- input validation ---
            if window.ndim != 2 or window.shape[1] != 6:
                raise ValueError("Each window must have shape (n_samples, 6)")

            hand_A = window[:, :3]
            hand_N = window[:, 3:]

            # --- vector magnitude ---
            vm_A = np.linalg.norm(hand_A, axis=1)
            vm_N = np.linalg.norm(hand_N, axis=1)

            # remove gravity (assumes unit=g)
            vm_A = np.maximum(vm_A - 1, 0)
            vm_N = np.maximum(vm_N - 1, 0)

            vm_A = np.nan_to_num(vm_A)
            vm_N = np.nan_to_num(vm_N)

            # --- total (note: signed sum) ---
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
                a = np.nan_to_num(a)
                m = np.nan_to_num(m)

                for k, v in self.summarize_signal(a).items():
                    feat[f"{name}_active_{k}"] = v
                for k, v in self.summarize_signal(m).items():
                    feat[f"{name}_mirror_{k}"] = v

            # ---------- BILATERAL ----------
            for name, (a, m) in signals.items():
                a = np.nan_to_num(a)
                m = np.nan_to_num(m)

                diff = a - m

                sum_a = float(np.sum(a))
                sum_m = float(np.sum(m))

                feat[f"{name}_bilateral_mean_diff"] = float(np.mean(diff))
                feat[f"{name}_bilateral_abs_mean_diff"] = float(np.mean(np.abs(diff)))

                feat[f"{name}_bilateral_sum_ratio"] = self.safe_div(sum_m, sum_a)

                feat[f"{name}_bilateral_asymmetry"] = self.safe_div(
                    (sum_a - sum_m),
                    (sum_a + sum_m)
                )

                feat[f"{name}_bilateral_corr0"] = self.safe_corr(a, m)

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

            # store feature names once
            if not hasattr(self, "feature_names_"):
                self.feature_names_ = list(feat.keys())

            all_features.append(list(feat.values()))

        # ---------- FINAL SAFETY ----------
        features = np.array(all_features, dtype=float)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features