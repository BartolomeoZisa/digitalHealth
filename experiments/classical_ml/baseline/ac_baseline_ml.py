"""
Mirror Movements classification - first iteration
-------------------------------------------------

This script implements an early classification pipeline for mirror movements
using Activity Count features extracted from accelerometer signals of both hands.

Pipeline:
1. Load TD and UCP CSV files.
2. Compute Activity Count (AC) from the 3-axis accelerometer signal:
       AC = sqrt(Axis1^2 + Axis2^2 + Axis3^2)
3. Split each session into fixed temporal windows.
4. Extract handcrafted features from each window, including:
   - mean and standard deviation for each hand
   - mean difference and absolute mean difference
   - standard deviation difference
   - ratio between the two hands
   - correlation between hand signals
   - signal energy
   - absolute difference statistics
   - max and min differences
5. Train classical classifiers on window-based features.
6. Evaluate performance with Leave-One-Subject-Out (LOSO) validation.
7. Aggregate window-level predictions at subject level.

Models used:
- Logistic Regression
- SVM
- KNN
- Random Forest

Goal:
Classify subjects as TD vs UCP using window-based Activity Count features.
"""



import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mode
import matplotlib.pyplot as plt

# -----------------------
# 1. Load data
# -----------------------
ucp = pd.read_csv("bbt_AC_anon.csv")
healthy = pd.read_csv("bbt_AC_sani_anon.csv")

ucp["label"] = 1
healthy["label"] = 0

data = pd.concat([ucp, healthy], ignore_index=True)

# -----------------------
# 2. Activity Count
# -----------------------
data["AC"] = np.sqrt(data["Axis1"]**2 + data["Axis2"]**2 + data["Axis3"]**2)

# -----------------------
# 3. Feature extraction LEFT vs RIGHT
# -----------------------
window_size = 10

features = []
labels = []
groups = []

for subject in data["id"].unique():
    subject_data = data[data["id"] == subject]
    
    hands = subject_data["hand"].unique()
    if len(hands) < 2:
        continue
    
    hand1 = subject_data[subject_data["hand"] == hands[0]]["AC"].values
    hand2 = subject_data[subject_data["hand"] == hands[1]]["AC"].values
    
    label = subject_data["label"].iloc[0]
    
    min_len = min(len(hand1), len(hand2))
    
    for i in range(0, min_len - window_size, window_size):
        w1 = hand1[i:i+window_size]
        w2 = hand2[i:i+window_size]
        
        # Safe correlation
        if np.std(w1) == 0 or np.std(w2) == 0:
            corr = 0
        else:
            corr = np.corrcoef(w1, w2)[0,1]
        
        ratio = np.mean(w1) / (np.mean(w2) + 1e-5)
        ratio = np.clip(ratio, 0, 10)
        
        feat = [
            np.mean(w1),
            np.mean(w2),
            np.std(w1),
            np.std(w2),
            np.mean(w1) - np.mean(w2),
            abs(np.mean(w1) - np.mean(w2)),
            np.std(w1) - np.std(w2),
            ratio,
            corr,
            np.log(np.sum(w1**2) + 1),
            np.log(np.sum(w2**2) + 1),
            np.mean(np.abs(w1 - w2)),
            np.std(np.abs(w1 - w2)),
            np.max(w1) - np.max(w2),
            np.min(w1) - np.min(w2)
        ]
        
        features.append(feat)
        labels.append(label)
        groups.append(subject)

X = np.array(features)
y = np.array(labels)
groups = np.array(groups)

print("Dataset shape:", X.shape)

# -----------------------
# 4. Clean data
# -----------------------
X = np.nan_to_num(X)
X = np.clip(X, -10, 10)

# -----------------------
# 5. Standardization
# -----------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------
# 6. Models
# -----------------------
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "SVM": SVC(class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced")
}

# -----------------------
# 7. Leave-One-Patient-Out
# -----------------------
logo = LeaveOneGroupOut()

results = {name: {"acc": [], "prec": [], "rec": [], "f1": []} for name in models}

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_test = groups[test_idx]

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_windows = model.predict(X_test)

        # Majority vote per subject
        subj_pred = {}
        subj_true = {}

        for s in np.unique(g_test):
            subj_pred[s] = round(np.mean(y_pred_windows[g_test == s]))
            subj_true[s] = y_test[g_test == s][0]

        y_true_final = list(subj_true.values())
        y_pred_final = list(subj_pred.values())

        results[name]["acc"].append(accuracy_score(y_true_final, y_pred_final))
        results[name]["prec"].append(precision_score(y_true_final, y_pred_final, zero_division=0))
        results[name]["rec"].append(recall_score(y_true_final, y_pred_final, zero_division=0))
        results[name]["f1"].append(f1_score(y_true_final, y_pred_final, zero_division=0))

# -----------------------
# 8. Print results
# -----------------------
print("\nFinal Results (LOPO - SUBJECT LEVEL):")
for name in models:
    print(f"\n{name}")
    print("Accuracy:", np.mean(results[name]["acc"]))
    print("Precision:", np.mean(results[name]["prec"]))
    print("Recall:", np.mean(results[name]["rec"]))
    print("F1-score:", np.mean(results[name]["f1"]))

# -----------------------
# 9. Feature importance
# -----------------------
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
rf.fit(X, y)

importances = rf.feature_importances_

feature_names = [
    "mean_hand1",
    "mean_hand2",
    "std_hand1",
    "std_hand2",
    "mean_diff",
    "abs_mean_diff",
    "std_diff",
    "ratio",
    "correlation",
    "energy1",
    "energy2",
    "mean_abs_diff",
    "std_abs_diff",
    "max_diff",
    "min_diff"
]

plt.figure()
plt.bar(feature_names, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()
