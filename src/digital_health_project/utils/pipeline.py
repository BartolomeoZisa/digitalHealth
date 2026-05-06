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
from digital_health_project.features.difference_assimmetry import PairedSignalMerger, TimeSeriesWindower, TimeseriesCleaner
from digital_health_project.utils.encoder import NumpyEncoder

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
    n_jobs = -1,
    column_names = [
            "Accelerometer X_A", "Accelerometer Y_A", "Accelerometer Z_A",
            "Accelerometer X_N", "Accelerometer Y_N", "Accelerometer Z_N"
    ],
):
    """
    Executes a nested cross-validation classification pipeline for time-series data.

    This function loads two classes of data (TD and UCP), merges them, applies 
    windowing/segmentation, and performs a GridSearch nested within a Cross-Validation 
    loop to ensure unbiased performance estimation and hyperparameter optimization.

    Args:
        td_path (str): File path to the 'Typically Developing' (class 0) CSV data.
        ucp_path (str): File path to the 'Unpaired/Clinical' (class 1) CSV data.
        pipeline_steps (list): List of (name, transform) tuples for the sklearn Pipeline.
        param_grid (list/dict): Dictionary or list of dictionaries with parameters 
            to try during the GridSearch.
        inner_cv (iter, optional): Cross-validation generator for the hyperparameter 
            search. Defaults to GroupKFold(n_splits=5).
        outer_cv (iter, optional): Cross-validation generator for the performance 
            estimation. Defaults to GroupKFold(n_splits=5).
        scoring (list, optional): List of sklearn-compatible metric strings. 
            Defaults to ['accuracy', 'f1', 'precision', 'recall'].
        refit_metric (str): The metric used to identify the best model in GridSearch.
            Defaults to 'f1'.
        window_size (int): Size of the sliding window (in samples). If set to -1, 
            the pipeline skips windowing and uses the full signal. Defaults to 240.
        step_size (int): The stride/overlap between windows. Defaults to 120.
        save_dir (str): Root directory for saving results and logs. Defaults to 'results'.
        experiment_name (str): Label for the current run, used in file naming. 
            Defaults to 'sktime_clustering_exp'.
        n_jobs (int): Number of CPU cores to use. -1 uses all available. Defaults to -1.
        column_names (list): List of sensor axis names to be processed.

    Returns:
        None: Results are saved to disk as CSV and JSON files in a timestamped 
            subdirectory within `save_dir`.

    Notes:
        - The function enforces 'patient-aware' splitting using `groups` to prevent 
          data leakage (windows from the same patient will not be split across 
          train/test sets).
        - It includes a compatibility wrapper for `cross_validate` to handle 
          varying scikit-learn version requirements for `fit_params`.
    """

         
    save_dir = os.path.join(save_dir, f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    if inner_cv is None: inner_cv = GroupKFold(n_splits=5)
    if outer_cv is None: outer_cv = GroupKFold(n_splits=5)
    
    # We can safely keep roc_auc now that the classifier tag is fixed
    if scoring is None: scoring = ['accuracy', 'f1', 'precision', 'recall']

    try:
        td, ucp = pd.read_csv(td_path), pd.read_csv(ucp_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}"); return

    td['label'], ucp['label'] = 0, 1
    full_df = pd.concat([td, ucp], ignore_index=True)

    merger = PairedSignalMerger()
    df_merged = merger.transform(full_df)

    if(window_size == -1):
        cleaner = TimeseriesCleaner(column_names=column_names)
        X_raw, y, groups = cleaner.transform(df_merged)
    else:    
        windower = TimeSeriesWindower(window_size=window_size, step_size=step_size, column_names=column_names)
        X_raw, y, groups = windower.transform(df_merged)

    #y = np.array(y).astype(np.int64)

    print(f"[{experiment_name}] Paired Dataset: {len(X_raw)} windows from {len(np.unique(groups))} patients.")

    pipeline = Pipeline(pipeline_steps)

    step_fit_params = {}
    for step_name, step_obj in pipeline_steps:
        if hasattr(step_obj, 'train_split'):
            step_fit_params[f"{step_name}__groups"] = groups


    grid_search = GridSearchCV(
        pipeline, param_grid, cv=inner_cv, 
        scoring=scoring, refit=refit_metric, n_jobs=n_jobs,
        error_score='raise', verbose = 2
    )

    print(f"Evaluating with Nested CV ({outer_cv.n_splits} splits)...")
    
    cv_fit_params = {'groups': groups, **step_fit_params}

    try:
        cv_results = cross_validate(
            grid_search, X_raw, y, groups=groups, cv=outer_cv, 
            scoring=scoring, return_train_score=False,
            params=cv_fit_params
        )
    except TypeError:
        cv_results = cross_validate(
            grid_search, X_raw, y, groups=groups, cv=outer_cv, 
            scoring=scoring, return_train_score=False,
            fit_params=cv_fit_params
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
