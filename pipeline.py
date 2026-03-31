import os
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score
import warnings
from pandas.errors import PerformanceWarning
# This silences the specific fragmentation warning from Pandas
warnings.filterwarnings('ignore', category=PerformanceWarning)


from sktime.clustering.k_means import TimeSeriesKMeans

# ==========================================
# 1. PAIRED WINDOWING
# ==========================================
class PairedSignalWindower:
    def __init__(self, window_size=240, step_size=120):
        self.window_size = window_size
        self.step_size = step_size

    def transform(self, df):
        X_paired, y_labels, groups = [], [], []
        
        for (pid, sess), session_df in df.groupby(['id', 'session']):
            dx_data = session_df[session_df['hand'] == 'dx'][['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']].values
            sx_data = session_df[session_df['hand'] == 'sx'][['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']].values
            
            min_len = min(len(dx_data), len(sx_data))
            if min_len < self.window_size:
                continue
                
            label = session_df['label'].iloc[0]
            
            for i in range(0, min_len - self.window_size, self.step_size):
                win_dx = dx_data[i : i + self.window_size]
                win_sx = sx_data[i : i + self.window_size]
                
                X_paired.append(np.hstack([win_dx, win_sx]))
                y_labels.append(label)
                groups.append(pid)
                
        # FIXED: Ensure y is explicitly integers so Sklearn doesn't assume Regression
        return np.array(X_paired), np.array(y_labels, dtype=int), np.array(groups)

# ==========================================
# 2. INTER-HAND PREPROCESSING
# ==========================================
class InterHandProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, mode='asymmetry_index'):
        self.mode = mode 

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed = []
        for window in X:
            mag_dx = np.sqrt(np.sum(np.square(window[:, :3]), axis=1))
            mag_sx = np.sqrt(np.sum(np.square(window[:, 3:]), axis=1))
            
            if self.mode == 'diff':
                res = mag_dx - mag_sx
            elif self.mode == 'asymmetry_index':
                res = ((mag_dx - mag_sx) / (mag_dx + mag_sx + 1e-9)) * 100
            else:
                res = np.column_stack([mag_dx, mag_sx])
            
            processed.append(res.reshape(-1, 1) if res.ndim == 1 else res)
            
        return np.array(processed)

# ==========================================
# 3. SKTIME DATA FORMATTER
# ==========================================
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested

class SktimeFormatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return np.transpose(X, (0, 2, 1))

# ==========================================
# 4. SKTIME CLUSTERING-TO-CLASSIFICATION WRAPPER
# ==========================================
class AlignedTimeSeriesKMeans(ClassifierMixin, BaseEstimator):
    
    # FIXED: Explicitly tell Scikit-Learn this pipeline step is a CLASSIFIER
    # This stops it from throwing the "Got a regressor with response_method=predict_proba" error
    _estimator_type = "classifier" 

    def __init__(self, n_clusters=2, init_algorithm='kmeans++', metric='euclidean', 
                 distance_params=None, averaging_method='mean', random_state=42):
        self.n_clusters = n_clusters
        self.init_algorithm = init_algorithm
        self.metric = metric
        self.distance_params = distance_params
        self.averaging_method = averaging_method
        self.random_state = random_state

    def fit(self, X, y):
        self.clusterer_ = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            init_algorithm=self.init_algorithm,
            metric=self.metric,                    
            distance_params=self.distance_params,  
            averaging_method=self.averaging_method,
            random_state=self.random_state
        )
        self.clusterer_.fit(X)
        
        # Align labels
        preds = self.clusterer_.predict(X)
        if accuracy_score(y, preds) < 0.5:
            self.flip_labels_ = True
        else:
            self.flip_labels_ = False
            
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        preds = self.clusterer_.predict(X)
        if self.flip_labels_:
            preds = 1 - preds
        return preds

    def predict_proba(self, X):
        # Generates binary probability outputs [0.0, 1.0] to satisfy ROC-AUC requirements
        preds = self.predict(X)
        probs = np.zeros((len(preds), len(self.classes_)))
        probs[np.arange(len(preds)), preds] = 1.0
        return probs


from sktime.classification.distance_based import ShapeDTW

# ==========================================
# 4. SHAPE-DTW CLASSIFIER WRAPPER
# ==========================================
# ==========================================
# 4. SHAPE-DTW CLASSIFIER WRAPPER (FIXED)
# ==========================================
class ShapeDTWClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, n_neighbors=1, shape_descriptor_function='raw'): # Removed subsequence_distance
        self.n_neighbors = n_neighbors
        self.shape_descriptor_function = shape_descriptor_function

    def fit(self, X, y):
        # We pass only the arguments accepted by sktime's ShapeDTW
        self.clf_ = ShapeDTW(
            n_neighbors=self.n_neighbors,
            shape_descriptor_function=self.shape_descriptor_function,
        )
        self.clf_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.clf_.predict(X)

    def predict_proba(self, X):
        # ShapeDTW might not implement predict_proba natively depending on version
        # If it fails, use the same logic as your KMeans wrapper
        try:
            return self.clf_.predict_proba(X)
        except AttributeError:
            preds = self.predict(X)
            probs = np.zeros((len(preds), len(self.classes_)))
            probs[np.arange(len(preds)), preds.astype(int)] = 1.0
            return probs

# ==========================================
# HELPER: Numpy to JSON Encoder
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ==========================================
# 5. MAIN PIPELINE EXECUTION
# ==========================================
def run_classification_pipeline(
    td_path: str, 
    ucp_path: str,
    pipeline_steps: list,
    param_grid: list,     
    inner_cv = None,
    outer_cv = None,
    scoring: list = None,
    refit_metric: str = 'f1',
    window_size: int = 240,
    step_size: int = 120,
    save_dir: str = 'results',
    experiment_name: str = 'sktime_clustering_exp',
    n_jobs = -1
):
    os.makedirs(save_dir, exist_ok=True)

    if inner_cv is None: inner_cv = GroupKFold(n_splits=5)
    if outer_cv is None: outer_cv = GroupKFold(n_splits=5)
    
    # We can safely keep roc_auc now that the classifier tag is fixed
    if scoring is None: scoring = ['accuracy', 'f1', 'roc_auc']

    try:
        td, ucp = pd.read_csv(td_path), pd.read_csv(ucp_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}"); return

    td['label'], ucp['label'] = 0, 1
    full_df = pd.concat([td, ucp], ignore_index=True)

    windower = PairedSignalWindower(window_size=window_size, step_size=step_size)
    X_raw, y, groups = windower.transform(full_df)
    print(f"[{experiment_name}] Paired Dataset: {len(X_raw)} windows from {len(np.unique(groups))} patients.")

    pipeline = Pipeline(pipeline_steps)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=inner_cv, 
        scoring=scoring, refit=refit_metric, n_jobs=-n_jobs,
        error_score='raise', verbose = 2
    )

    print(f"Evaluating with Nested CV ({outer_cv.n_splits} splits)...")
    
    try:
        cv_results = cross_validate(
            grid_search, X_raw, y, groups=groups, cv=outer_cv, 
            scoring=scoring, return_train_score=False,
            params={'groups': groups} 
        )
    except TypeError:
        cv_results = cross_validate(
            grid_search, X_raw, y, groups=groups, cv=outer_cv, 
            scoring=scoring, return_train_score=False,
            fit_params={'groups': groups}
        )

    grid_search.fit(X_raw, y, groups=groups)

    pd.DataFrame(cv_results).to_csv(os.path.join(save_dir, f"{experiment_name}_nested_cv.csv"), index=False)
    pd.DataFrame(grid_search.cv_results_).to_csv(os.path.join(save_dir, f"{experiment_name}_grid_search.csv"), index=False)

    summary = {
        "experiment_name": experiment_name,
        "window_size": window_size,
        "step_size": step_size,
        "dataset_windows": len(X_raw),
        "unique_patients": len(np.unique(groups)),
        "best_overall_parameters": grid_search.best_params_,
        "best_overall_score": grid_search.best_score_,
        "nested_cv_metrics": {
            metric: {"mean": np.mean(cv_results[f"test_{metric}"]), "std": np.std(cv_results[f"test_{metric}"])} 
            for metric in scoring
        }
    }
    
    with open(os.path.join(save_dir, f"{experiment_name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4, cls=NumpyEncoder)

    print("\n" + "="*45)
    print(f"RESULTS FOR: {experiment_name}")
    for metric in scoring:
        m = np.mean(cv_results[f"test_{metric}"])
        s = np.std(cv_results[f"test_{metric}"])
        print(f"Test {metric.upper()}: {m:.4f} ± {s:.4f}")
    print("="*45)
    print(f"Best Config (Full Data): {grid_search.best_params_}\n")






if __name__ == "__main__":
    TD_PATH = 'data/bbt_RAW_TD_clean.csv'
    UCP_PATH = 'data/bbt_RAW_UCP_clean.csv'

    '''
    # --- RUN 1: K-MEANS (Standard Metrics) ---
    kmeans_pipeline = [
        ('inter_hand', InterHandProcessor()),
        ('sktime_formatter', SktimeFormatTransformer()),
        ('clf', AlignedTimeSeriesKMeans())
    ]
    
    kmeans_params = [{
        'inter_hand__mode': ['diff', 'asymmetry_index'],
        'clf__metric': ['euclidean', 'dtw'],
        'clf__init_algorithm': ['kmeans++', 'forgy'],
        'clf__n_clusters': [2]
    }]

    run_classification_pipeline(
        td_path=TD_PATH, ucp_path=UCP_PATH,
        pipeline_steps=kmeans_pipeline,
        param_grid=kmeans_params,
        experiment_name='KMeans_Baseline'
    )
    '''

    # --- RUN 2: ACTUAL SHAPE-DTW CLASSIFICATION ---
    shapedtw_pipeline = [
        ('inter_hand', InterHandProcessor()),
        ('sktime_formatter', SktimeFormatTransformer()),
        ('clf', ShapeDTWClassifier()) # Using the new wrapper
    ]

    shapedtw_params = [{
        'inter_hand__mode': ['diff', 'asymmetry_index'],
        'clf__shape_descriptor_function': ['raw', 'paa'],
    }]

    run_classification_pipeline(
        td_path=TD_PATH, ucp_path=UCP_PATH,
        pipeline_steps=shapedtw_pipeline,
        param_grid=shapedtw_params,
        experiment_name='ShapeDTW_Direct_Classifier',
        n_jobs=1
    )