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
from digital_health_project.utils.encoder import NumpyEncoder
from digital_health_project.features.difference_assimmetry import PairedSignalWindower, InterHandProcessor

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
         
    save_dir = os.path.join(save_dir, f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    if inner_cv is None: inner_cv = GroupKFold(n_splits=5)
    if outer_cv is None: outer_cv = GroupKFold(n_splits=5)
    
    # We can safely keep roc_auc now that the classifier tag is fixed
    if scoring is None: scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

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
        scoring=scoring, refit=refit_metric, n_jobs=n_jobs,
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

    #time_stamped_di
    summary = {
        "experiment_name": experiment_name,
        "window_size": window_size,
        "step_size": step_size,
        "dataset_windows": len(X_raw),
        "unique_patients": len(np.unique(groups)),
        "best_overall_parameters": grid_search.best_params_,
        "best_validation_score": grid_search.best_score_,
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
