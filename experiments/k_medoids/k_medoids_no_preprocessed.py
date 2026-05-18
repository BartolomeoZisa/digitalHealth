from sklearn.pipeline import Pipeline
from digital_health_project.models.models import AlignedTimeSeriesKMeans, AlignedTimeSeriesKMeans, AlignedTimeSeriesKMedoids
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.difference_assimmetry import InterHandProcessor, SktimeFormatTransformer


TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


k_medoids_pipeline = [
    ('sktime_formatter', SktimeFormatTransformer()),
    ('clf', AlignedTimeSeriesKMedoids())
]

k_medoids_params = [{
    'clf__metric': ['euclidean', 'dtw'],
    'clf__init_algorithm': ['forgy', 'random'],
    'clf__n_clusters': [2]
}]

run_classification_pipeline(
    td_path=TD_PATH, ucp_path=UCP_PATH,
    pipeline_steps=k_medoids_pipeline,
    param_grid=k_medoids_params,
    save_dir='results/k_medoids_no_preprocess',
    experiment_name='KMedoids_no_preprocess_Accuracy'
)
