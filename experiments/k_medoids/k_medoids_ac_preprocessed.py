from sklearn.pipeline import Pipeline
from digital_health_project.models.models import AlignedTimeSeriesKMeans, AlignedTimeSeriesKMeans, AlignedTimeSeriesKMedoids
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.difference_assimmetry import InterHandProcessor, SktimeFormatTransformer


TD_PATH = 'data/td/bbt_td_1sec_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_1sec_anon.csv'


k_medoids_pipeline = [
    ('inter_hand', InterHandProcessor()),
    ('sktime_formatter', SktimeFormatTransformer()),
    ('clf', AlignedTimeSeriesKMedoids())
]

k_medoids_params = [{
    'inter_hand__mode': ['diff', 'asymmetry_index'],
    'clf__metric': ['euclidean', 'dtw'],
    'clf__init_algorithm': ['forgy', 'random'],
    'clf__n_clusters': [2]
}]

run_classification_pipeline(
    td_path=TD_PATH, ucp_path=UCP_PATH,
    pipeline_steps=k_medoids_pipeline,
    param_grid=k_medoids_params,
    window_size=-1,
    save_dir='results/k_medoids_ac_preprocess',
    experiment_name='KMedoids_preprocess_ac_Accuracy',
    column_names=["Axis1_A", "Axis2_A", "Axis3_A", "Axis1_N", "Axis2_N", "Axis3_N"]
)
