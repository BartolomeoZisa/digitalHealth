import os
import json
import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import accuracy_score
from pandas.errors import PerformanceWarning

from sktime.clustering.k_means import TimeSeriesKMeans

warnings.filterwarnings('ignore', category=PerformanceWarning)


# ==========================================
# 1. PAIRED WINDOWING (SKLEARN-COMPATIBLE)
# ==========================================
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PairedSignalMerger(BaseEstimator, TransformerMixin):
    """
    Merges active and inactive hand signals into a single row per timestamp.
    Input: Long-format DataFrame
    Output: Merged DataFrame with suffixes _A (active) and _N (non-active)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()
        df = df.sort_values(['id', 'session', 'datetime'])
        
        # Identify hand roles
        df["is_active"] = df["session"] == df["hand_type"]

        merged_list = []

        for (pid, sess), session_df in df.groupby(['id', 'session']):
            active_df = session_df[session_df["is_active"]]
            inactive_df = session_df[~session_df["is_active"]]

            if active_df.empty or inactive_df.empty:
                continue

            merged = pd.merge(
                active_df,
                inactive_df,
                on="datetime",
                suffixes=("_A", "_N"),
                how="inner"
            ).sort_values("datetime")
            
            # Keep necessary ID/Label info for the next step
            merged_list.append(merged)

        return pd.concat(merged_list, ignore_index=True)


class TimeSeriesWindower(BaseEstimator, TransformerMixin):
    """
    Sliding window extractor. 
    Expects a DataFrame already merged by PairedSignalMerger.
    """
    def __init__(self, window_size=240, step_size=120, column_names=None):
        self.window_size = window_size
        self.step_size = step_size
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        X_windows = []
        y_labels = []
        groups = []

        # Group by the merged session keys
        # Note: suffixes from Merger are id_A and session_A
        for (pid, sess), session_df in df.groupby(['id_A', 'session_A']):
            if len(session_df) < self.window_size:
                continue

            # Extract 6-channel data (3 active, 3 inactive)
            if self.column_names is None:
                self.column_names = [
                    "Accelerometer X_A", "Accelerometer Y_A", "Accelerometer Z_A",
                    "Accelerometer X_N", "Accelerometer Y_N", "Accelerometer Z_N"
                ]
            signals = session_df[self.column_names].values
            
            label = int(session_df["label_A"].iloc[0])

            for i in range(0, len(session_df) - self.window_size + 1, self.step_size):
                window = signals[i : i + self.window_size]
                X_windows.append(window)
                y_labels.append(label)
                groups.append(pid)

        return np.array(X_windows), np.array(y_labels), np.array(groups)

    
class TimeseriesCleaner(BaseEstimator, TransformerMixin):
    """
    Selects specific columns and returns a grouped NumPy array.
    """
    def __init__(self, column_names=None):
        # Default columns if none provided
        self.column_names = column_names or [
            "Accelerometer X_A", "Accelerometer Y_A", "Accelerometer Z_A",
            "Accelerometer X_N", "Accelerometer Y_N", "Accelerometer Z_N"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # Use a list comprehension for a cleaner, faster loop
        results = [
            (sess_df[self.column_names].values, int(sess_df["label_A"].iloc[0]), pid)
            for (pid, sess), sess_df in df.groupby(['id_A', 'session_A'])
        ]
        
        # Unzip the results into three separate arrays
        X_windows, y_labels, groups = zip(*results)
        
        return np.array(X_windows), np.array(y_labels), np.array(groups)
        




# ==========================================
# 2. INTER-HAND PROCESSOR
# ==========================================
class InterHandProcessor(BaseEstimator, TransformerMixin):
    """
    Computes inter-hand derived signals from paired accelerometer windows.

    Modes:
    - diff: absolute difference of magnitudes
    - asymmetry_index: normalized asymmetry measure
    - default: returns both magnitudes
    """

    def __init__(self, mode='asymmetry_index'):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : np.ndarray
            Shape: (samples, time, features)
            Features expected: [L_x, L_y, L_z, R_x, R_y, R_z]

        Returns
        -------
        np.ndarray
            Processed inter-hand representation per window
        """

        processed = []

        for window in X:
            # Compute vector magnitude per hand
            mag_L = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
            mag_R = np.sqrt(np.sum(window[:, 3:] ** 2, axis=1))

            if self.mode == 'diff':
                res = mag_L - mag_R

            elif self.mode == 'asymmetry_index':
                #start from enmo to remove gravity
                mag_L = np.maximum(0, mag_L - 1)
                mag_R = np.maximum(0, mag_R - 1)

                res = ((mag_L - mag_R) / (mag_L + mag_R + 1e-9)) * 100

            else:
                res = np.column_stack([mag_L, mag_R])

            processed.append(
                res.reshape(-1, 1) if res.ndim == 1 else res
            )

        return np.array(processed)


# ==========================================
# 3. SKTIME FORMATTER
# ==========================================
class SktimeFormatTransformer(BaseEstimator, TransformerMixin):
    """
    Reorders tensor dimensions to match sktime expected format.

    Converts:
    (samples, time, features)
        -> (samples, features, time)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.transpose(X, (0, 2, 1))


# ==========================================
# 4. KMEANS WRAPPER (CLUSTERING MODEL)
# ==========================================
class AlignedTimeSeriesKMeans(BaseEstimator):
    """
    Wrapper around sktime TimeSeriesKMeans for sklearn pipeline compatibility.
    """

    def __init__(self, n_clusters=2, metric="euclidean", init_algorithm="kmeans++"):
        self.n_clusters = n_clusters
        self.metric = metric
        self.init_algorithm = init_algorithm

        self.model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric=metric,
            init_algorithm=init_algorithm,
            random_state=42
        )

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.predict(X)
