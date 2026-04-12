from sklearn.pipeline import Pipeline
from digital_health_project.models.models import AlignedTimeSeriesKMeans
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.difference_assimmetry import InterHandProcessor, SktimeFormatTransformer, PairedSignalWindower
from sklearn.preprocessing import StandardScaler

TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


kmeans_pipeline = [
    ('scaler', StandardScaler()),
    ('sktime_formatter', SktimeFormatTransformer()),
    ('clf', AlignedTimeSeriesKMeans())
]

kmeans_params = [{
    'clf__metric': ['euclidean', 'dtw'],
    'clf__init_algorithm': ['kmeans++', 'forgy'],
    'clf__n_clusters': [2]
}]

run_classification_pipeline(
    td_path=TD_PATH, ucp_path=UCP_PATH,
    pipeline_steps=kmeans_pipeline,
    param_grid=kmeans_params,
    save_dir='results/k_means_normalization',
    experiment_name='KMeans_normalization'
)
