import os
import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PAIRED WINDOWING
# ==========================================
class PairedSignalWindower:
    def __init__(self, window_size=320, step_size=160):
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
                
        return np.array(X_paired), np.array(y_labels), np.array(groups)

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
# 3. FEATURE EXTRACTION
# ==========================================
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    
    def transform(self, X):
        features = []
        for window in X:
            win_feats = []
            for channel in range(window.shape[1]):
                sig = window[:, channel]
                win_feats.extend([
                    np.mean(sig), np.std(sig), np.max(sig), 
                    np.min(sig), np.sqrt(np.mean(sig**2))
                ])
            features.append(win_feats)
        return np.array(features)

# ==========================================
# HELPER: Numpy to JSON Encoder
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================
def run_classification_pipeline(
    td_path: str, 
    ucp_path: str,
    pipeline_steps: list = None,
    param_grid: dict = None,
    inner_cv = None,
    outer_cv = None,
    scoring: list = None,
    refit_metric: str = 'f1',
    window_size: int = 320,
    step_size: int = 160,
    save_dir: str = 'results',
    experiment_name: str = 'experiment_1'
):
    """
    Runs a highly configurable nested cross-validation pipeline and saves results.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Setup Defaults if None ---
    if pipeline_steps is None:
        pipeline_steps = [
            ('inter_hand', InterHandProcessor()),
            ('extractor', FeatureExtractor()), # Extractor needed to convert 3D to 2D for standard ML
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42))
        ]
    if param_grid is None:
        param_grid = {
            'inter_hand__mode': ['diff', 'asymmetry_index'],
            'clf__n_estimators': [100, 200]
        }
    if inner_cv is None: inner_cv = GroupKFold(n_splits=5)
    if outer_cv is None: outer_cv = GroupKFold(n_splits=5)
    if scoring is None: scoring = ['accuracy', 'f1', 'roc_auc']

    # --- Load Data ---
    try:
        td, ucp = pd.read_csv(td_path), pd.read_csv(ucp_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}"); return

    td['label'], ucp['label'] = 0, 1
    full_df = pd.concat([td, ucp], ignore_index=True)

    # --- Step 1: Paired Windowing ---
    windower = PairedSignalWindower(window_size=window_size, step_size=step_size)
    X_raw, y, groups = windower.transform(full_df)
    print(f"[{experiment_name}] Paired Dataset: {len(X_raw)} windows from {len(np.unique(groups))} patients.")

    # --- Step 2: Create Pipeline & Search ---
    pipeline = Pipeline(pipeline_steps)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=inner_cv, 
        scoring=scoring, refit=refit_metric, n_jobs=-1
    )

    # --- Step 3: Nested Cross-Validation (Outer loop) ---
    print(f"Evaluating with Nested CV ({outer_cv.n_splits} splits)...")
    cv_results = cross_validate(
        grid_search, X_raw, y, groups=groups, cv=outer_cv, 
        scoring=scoring, return_train_score=False
    )

    # --- Step 4: Final Fit (to get best overall parameters on all data) ---
    grid_search.fit(X_raw, y, groups=groups)

    # --- Step 5: Save Results ---
    # 5a. Save Nested CV scores to CSV
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(save_dir, f"{experiment_name}_nested_cv.csv"), index=False)

    # 5b. Save Full Grid Search results to CSV
    grid_df = pd.DataFrame(grid_search.cv_results_)
    grid_df.to_csv(os.path.join(save_dir, f"{experiment_name}_grid_search.csv"), index=False)

    # 5c. Compile Summary and save to JSON
    summary = {
        "experiment_name": experiment_name,
        "window_size": window_size,
        "step_size": step_size,
        "dataset_windows": len(X_raw),
        "unique_patients": len(np.unique(groups)),
        "best_overall_parameters": grid_search.best_params_,
        "best_overall_score": grid_search.best_score_,
        "nested_cv_metrics": {
            metric: {
                "mean": np.mean(cv_results[f"test_{metric}"]),
                "std": np.std(cv_results[f"test_{metric}"])
            } for metric in scoring
        }
    }
    
    with open(os.path.join(save_dir, f"{experiment_name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4, cls=NumpyEncoder)

    # --- Step 6: Print Quick Summary ---
    print("\n" + "="*45)
    print(f"RESULTS FOR: {experiment_name}")
    for metric in scoring:
        m = np.mean(cv_results[f"test_{metric}"])
        s = np.std(cv_results[f"test_{metric}"])
        print(f"Test {metric.upper()}: {m:.4f} ± {s:.4f}")
    print("="*45)
    print(f"Best Config (Full Data): {grid_search.best_params_}\n")


# ==========================================
# HOW TO USE THE NEW FUNCTION
# ==========================================
if __name__ == "__main__":
    
    # NOTE: Replace with your actual paths
    TD_PATH = 'data/bbt_RAW_TD_clean.csv'
    UCP_PATH = 'data/bbt_RAW_UCP_clean.csv'



    # ---------------------------------------------------------
    # EXPERIMENT 1: Default Settings (Random Forest)
    # ---------------------------------------------------------
    run_classification_pipeline(
        td_path=TD_PATH, 
        ucp_path=UCP_PATH,
        experiment_name='Exp1_RandomForest_Default'
    )


    # ---------------------------------------------------------
    # EXPERIMENT 2: Custom Pipeline (SVM), Custom Grid & Window
    # ---------------------------------------------------------
    svm_pipeline_steps = [
        ('inter_hand', InterHandProcessor()),
        ('extractor', FeatureExtractor()),
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42)) # SVM instead of RF
    ]

    svm_param_grid = {
        'inter_hand__mode': ['diff'],  # test just 'diff'
        'clf__C': [0.1, 1, 10],        # SVM params
        'clf__kernel': ['linear', 'rbf']
    }

    run_classification_pipeline(
        td_path=TD_PATH, 
        ucp_path=UCP_PATH,
        pipeline_steps=svm_pipeline_steps,
        param_grid=svm_param_grid,
        window_size=200,          # Try a smaller window
        step_size=100,
        inner_cv=GroupKFold(n_splits=3), # Custom CV splits
        outer_cv=GroupKFold(n_splits=3),
        scoring=['accuracy', 'f1'],      # Custom metrics
        refit_metric='f1',               # Metric to optimize
        save_dir='my_custom_results',
        experiment_name='Exp2_SVM_CustomWindow'
    )