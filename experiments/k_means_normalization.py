from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from digital_health_project.models.models import AlignedTimeSeriesKMeans
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.difference_assimmetry import (
    SktimeFormatTransformer
)

TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


# ----------------------------
# 3D-safe normalization
# ----------------------------
def z_norm_3d(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


kmeans_pipeline = [
    # convert raw dataframe → time series format first
    ('sktime_formatter', SktimeFormatTransformer()),

    # safe scaling for (n_samples, time, features)
    ('scaler', FunctionTransformer(z_norm_3d)),

    # clustering model
    ('clf', AlignedTimeSeriesKMeans())
]


kmeans_params = [{
    'clf__metric': ['euclidean', 'dtw'],
    'clf__init_algorithm': ['kmeans++', 'forgy'],
    'clf__n_clusters': [2]
}]


run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=kmeans_pipeline,
    param_grid=kmeans_params,
    save_dir='results/k_means_normalization',
    experiment_name='KMeans_normalization'
)