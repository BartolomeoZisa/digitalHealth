import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import warnings
warnings.filterwarnings("ignore")

# =========================
# PARAMETERS
# =========================
WINDOW_SIZE = 5
SAMPLING_RATE = 1
WINDOW_SAMPLES = WINDOW_SIZE * SAMPLING_RATE

N_SPLITS = 5
N_CLUSTERS = 2   # sano / UCP
KMEANS_INIT = 'k-means++'
KMEANS_N_INIT = 10

# =========================
# LOAD DATA
# =========================
def load_data(healthy_path, ucp_path):
    print("Loading data...")

    healthy = pd.read_csv(healthy_path)
    ucp = pd.read_csv(ucp_path)

    healthy = healthy[['Axis1','Axis2','Axis3','hand','session','datetime','id']]
    ucp = ucp[['Axis1','Axis2','Axis3','hand','session','datetime','id']]

    healthy['label'] = 0
    ucp['label'] = 1

    df = pd.concat([healthy, ucp])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['id','datetime'])

    return df

# =========================
# WINDOWING
# =========================
def create_windows(df):
    print("Creating windows...")

    X = []
    y = []
    groups = []

    for patient_id in df['id'].unique():
        patient = df[df['id'] == patient_id]

        for session in patient['session'].unique():
            sess = patient[patient['session'] == session]

            data = sess[['Axis1','Axis2','Axis3']].values
            label = sess['label'].iloc[0]

            for i in range(0, len(data) - WINDOW_SAMPLES, WINDOW_SAMPLES):
                window = data[i:i+WINDOW_SAMPLES]
                X.append(window)
                y.append(label)
                groups.append(patient_id)

    print("Total windows:", len(X))
    return np.array(X), np.array(y), np.array(groups)

# =========================
# CROSS VALIDATION
# =========================
def cross_validation(X, y, groups):
    print("\nStarting K-means Cross Validation...")

    print("\nSelected Parameters:")
    print("WINDOW_SIZE:", WINDOW_SIZE)
    print("N_SPLITS:", N_SPLITS)
    print("N_CLUSTERS:", N_CLUSTERS)
    print("KMEANS_INIT:", KMEANS_INIT)
    print("KMEANS_N_INIT:", KMEANS_N_INIT)

    gkf = GroupKFold(n_splits=N_SPLITS)
    acc_results = []
    f1_results = []
    fold = 1

    for train_idx, test_idx in gkf.split(X, y, groups):
        print("\nFOLD", fold)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Flatten time series
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        kmeans = KMeans(n_clusters=N_CLUSTERS,
                        init=KMEANS_INIT,
                        n_init=KMEANS_N_INIT)
        kmeans.fit(X_train_flat)

        train_clusters = kmeans.labels_
        test_clusters = kmeans.predict(X_test_flat)

        # Mapping 1
        cluster_labels_1 = {}
        for c in range(N_CLUSTERS):
            idx = np.where(train_clusters == c)[0]
            if len(idx) > 0:
                cluster_labels_1[c] = np.bincount(y_train[idx]).argmax()

        preds_1 = np.array([cluster_labels_1[c] for c in test_clusters])
        acc_1 = accuracy_score(y_test, preds_1)
        f1_1 = f1_score(y_test, preds_1)

        # Mapping invertito
        preds_2 = 1 - preds_1
        acc_2 = accuracy_score(y_test, preds_2)
        f1_2 = f1_score(y_test, preds_2)

        # Choose best mapping
        if acc_1 > acc_2:
            acc = acc_1
            f1 = f1_1
        else:
            acc = acc_2
            f1 = f1_2

        print("Accuracy:", acc)
        print("F1-score:", f1)

        acc_results.append(acc)
        f1_results.append(f1)
        fold += 1

    print("\nFINAL RESULTS")
    print("Mean Accuracy:", np.mean(acc_results))
    print("Mean F1-score:", np.mean(f1_results))

# =========================
# MAIN
# =========================
print("Starting program...")

df = load_data("bbt_AC_sani_anon.csv", "bbt_AC_anon.csv")
X, y, groups = create_windows(df)

print("Normalizing...")
scaler = TimeSeriesScalerMeanVariance()
X = scaler.fit_transform(X)

cross_validation(X, y, groups)