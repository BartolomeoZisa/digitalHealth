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
class PairedSignalWindower(BaseEstimator, TransformerMixin):
    """
    Converts raw multi-hand time-series data into paired sliding windows.

    This transformer:
    - Aligns left (L) and right (R) hand signals by timestamp
    - Builds fixed-length overlapping windows
    - Produces paired feature tensors per window
    - Returns labels and subject/group IDs for grouped CV
    """

    def __init__(self, window_size=240, step_size=120):
        """
        Parameters
        ----------
        window_size : int
            Number of time steps per window.
        step_size : int
            Step size for sliding window.
        """
        self.window_size = window_size
        self.step_size = step_size

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Must contain:
            - id (subject)
            - session
            - datetime
            - hand (L/R)
            - Accelerometer X/Y/Z (per hand)
            - label

        Returns
        -------
        X_paired : np.ndarray
            Shape: (n_windows, window_size, 6)
        y_labels : np.ndarray
            Window labels
        groups : np.ndarray
            Subject IDs for grouped cross-validation
        """

        X_paired, y_labels, groups = [], [], []

        df = df.sort_values(['id', 'session', 'datetime'])

        for (pid, sess), session_df in df.groupby(['id', 'session']):

            left = session_df[session_df['hand'] == 'L']
            right = session_df[session_df['hand'] == 'R']

            if left.empty or right.empty:
                continue

            # Align both hands on timestamp
            merged = pd.merge(
                left,
                right,
                on='datetime',
                suffixes=('_L', '_R'),
                how='inner'
            ).sort_values('datetime')

            if len(merged) < self.window_size:
                continue

            # Extract accelerometer signals
            L_data = merged[
                ['Accelerometer X_L', 'Accelerometer Y_L', 'Accelerometer Z_L']
            ].values

            R_data = merged[
                ['Accelerometer X_R', 'Accelerometer Y_R', 'Accelerometer Z_R']
            ].values

            label = merged['label_L'].iloc[0]

            # Sliding window creation
            for i in range(0, len(merged) - self.window_size + 1, self.step_size):
                win_L = L_data[i:i + self.window_size]
                win_R = R_data[i:i + self.window_size]

                # Concatenate left and right hand features
                X_paired.append(np.hstack([win_L, win_R]))
                y_labels.append(label)
                groups.append(pid)

        return np.array(X_paired), np.array(y_labels), np.array(groups)


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


# ==========================================
# 5. PIPELINE RUNNER
# ==========================================
def run_classification_pipeline(
    td_path,
    ucp_path,
    pipeline_steps,
    param_grid,
    save_dir,
    experiment_name
):
    """
    Runs grouped cross-validation pipeline on paired time-series data.

    Steps:
    1. Load datasets (TD + UCP)
    2. Generate paired sliding windows
    3. Build sklearn pipeline
    4. Evaluate using GroupKFold CV
    5. Save results to disk

    Parameters
    ----------
    td_path : str
        Path to TD dataset CSV
    ucp_path : str
        Path to UCP dataset CSV
    pipeline_steps : list
        sklearn Pipeline steps (name, transformer/model)
    param_grid : dict
        (Currently unused in this function, reserved for future tuning)
    save_dir : str
        Directory to store results
    experiment_name : str
        Identifier for experiment logging
    """

    os.makedirs(save_dir, exist_ok=True)

    df_td = pd.read_csv(td_path)
    df_ucp = pd.read_csv(ucp_path)

    df = pd.concat([df_td, df_ucp], ignore_index=True)

    print(f"[{experiment_name}] Dataset loaded: {df.shape}")

    # STEP 1: WINDOWING (OUTSIDE GRID SEARCH!)
    windower = PairedSignalWindower()
    X, y, groups = windower.transform(df)

    print(f"[{experiment_name}] Paired Dataset: {len(X)} windows from {len(np.unique(groups))} patients.")

    if len(X) == 0:
        raise ValueError("No windows generated. Check hand labels and timestamp alignment.")

    # STEP 2: PIPELINE
    pipe = Pipeline(pipeline_steps)

    # STEP 3: CROSS VALIDATION
    cv = GroupKFold(n_splits=5)

    print("Evaluating with Nested CV (5 splits)...")

    results = cross_validate(
        pipe,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="accuracy",
        return_train_score=True
    )

    # STEP 4: SAVE RESULTS
    output = {
        "experiment": experiment_name,
        "accuracy_mean": float(np.mean(results["test_score"])),
        "accuracy_std": float(np.std(results["test_score"]))
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=4)

    print(output)