import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

from joblib import Parallel, delayed
import time



# =========================
# 0. SET PYTORCH THREADS
# =========================

device = "cuda" 

torch.set_num_threads(1)  # important for joblib parallelism

# =========================
# 1. LOAD DATA
# =========================
df_healthy = pd.read_csv("parquet_converted_formats/CSV/bbt_RAW_anon.csv")
df_unhealthy = pd.read_csv("parquet_converted_formats/CSV/bbt_RAW_sani_anon.csv")

df_healthy["label"] = 0
df_unhealthy["label"] = 1

df = pd.concat([df_healthy, df_unhealthy], ignore_index=True)
df = df.sort_values(by=["id", "session", "hand", "datetime"])

# =========================
# 2. WINDOWING (DUAL HANDS)
# =========================
def create_windows_dual_hands(df, window_size, stride):
    X, y, groups = [], [], []

    # Group by patient + session
    for (pid, session), group in df.groupby(["id", "session"]):
        # Separate left and right hand
        left = group[group['hand'] == 'sx'][['Accelerometer X','Accelerometer Y','Accelerometer Z']].values
        right = group[group['hand'] == 'dx'][['Accelerometer X','Accelerometer Y','Accelerometer Z']].values

        # Make sure both hands have same length
        min_len = min(len(left), len(right))
        left, right = left[:min_len], right[:min_len]

        data = np.hstack([left, right])  # shape (timesteps, 6)
        label = group['label'].iloc[0]

        # Sliding windows
        for i in range(0, len(data) - window_size, stride):
            X.append(data[i:i+window_size])
            y.append(label)
            groups.append(pid)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(groups)

# =========================
# 3. MODEL
# =========================
class LSTMNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.dropout(hn[-1])
        x = self.fc(x)
        return x

# =========================
# 4. AGGREGATION
# =========================
def aggregate_predictions(y_pred, groups):
    df_pred = pd.DataFrame({"pred": y_pred.flatten(), "group": groups})
    return df_pred.groupby("group")["pred"].mean().round()

# =========================
# 5. PARAM GRID
# =========================
param_grid = {
    "window_size": [80, 160],
    "stride": [40],
    "hidden_size": [32, 64],
    "dropout": [0.2, 0.4],
    "lr": [0.001],
    "batch_size": [16],
    "max_epochs": [20]
}

# =========================
# 6. INNER GRID SEARCH FUNCTION (parallel)
# =========================
def evaluate_params(params, df_train, inner_cv):
    print(f"started evaluate {params}")
    start = time.time()  # record start time
    X_all, y_all, groups_all = create_windows_dual_hands(df_train, params["window_size"], params["stride"])
    inner_scores = []

    for train_idx, val_idx in inner_cv.split(X_all, y_all, groups_all):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]
        groups_val = groups_all[val_idx]

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, 6)).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, 6)).reshape(X_val.shape)

        # Model
        net = NeuralNetClassifier(
            LSTMNet,
            module__input_size=6,
            module__hidden_size=params["hidden_size"],
            module__dropout=params["dropout"],
            max_epochs=params["max_epochs"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            optimizer=torch.optim.Adam,
            criterion=nn.BCEWithLogitsLoss,
            verbose=0,
            device= device
        )

        # Fit with correct target shape
        net.fit(X_train, y_train.reshape(-1, 1))

        # predictions
        y_pred = net.predict(X_val)

        # patient-level aggregation
        y_pred_patient = aggregate_predictions(y_pred, groups_val)
        y_true_patient = pd.Series(y_val, index=groups_val).groupby(level=0).first()

        score = accuracy_score(y_true_patient, y_pred_patient)
        inner_scores.append(score)
    end = time.time()  # record end time
    print(f"end evaluate {params} \n {start-end} s")
    return np.mean(inner_scores), params

# =========================
# 7. NESTED CROSS-VALIDATION
# =========================
outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=3)
outer_results = []

for fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(df, df["label"], groups=df["id"])):

    print(f"\n===== OUTER FOLD {fold+1} =====")
    df_train = df.iloc[outer_train_idx]
    df_test = df.iloc[outer_test_idx]

    # ---------------------
    # PARALLEL GRID SEARCH
    # ---------------------
    
    n_jobs = torch.cuda.device_count() if device == "cuda" else -1

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(params, df_train, inner_cv)
        for params in ParameterGrid(param_grid)
    )

    best_score, best_params = max(results, key=lambda x: x[0])
    print("Best params:", best_params)

    # ---------------------
    # FINAL TRAINING ON OUTER TRAIN
    # ---------------------
    X_train, y_train, groups_train = create_windows_dual_hands(
        df_train, best_params["window_size"], best_params["stride"])
    X_test, y_test, groups_test = create_windows_dual_hands(
        df_test, best_params["window_size"], best_params["stride"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 6)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, 6)).reshape(X_test.shape)

    net = NeuralNetClassifier(
        LSTMNet,
        module__input_size=6,
        module__hidden_size=best_params["hidden_size"],
        module__dropout=best_params["dropout"],
        max_epochs=best_params["max_epochs"],
        lr=best_params["lr"],
        batch_size=best_params["batch_size"],
        optimizer=torch.optim.Adam,
        criterion=nn.BCEWithLogitsLoss,
        verbose=0,
        device=device
    )

    net.fit(X_train, y_train.reshape(-1, 1))

    y_pred = net.predict(X_test)

    y_pred_patient = aggregate_predictions(y_pred, groups_test)
    y_true_patient = pd.Series(y_test, index=groups_test).groupby(level=0).first()

    test_score = accuracy_score(y_true_patient, y_pred_patient)
    print("Outer fold patient accuracy:", test_score)
    outer_results.append(test_score)

# =========================
# FINAL RESULT
# =========================
print("\n=========================")
print("FINAL MEAN ACCURACY:", np.mean(outer_results))
print("=========================")