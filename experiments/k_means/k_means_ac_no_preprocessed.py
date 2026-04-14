from sklearn.pipeline import Pipeline
from digital_health_project.models.models import AlignedTimeSeriesKMeans
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.difference_assimmetry import InterHandProcessor, SktimeFormatTransformer


TD_PATH = 'data/td/bbt_td_1sec_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_1sec_anon.csv'

kmeans_pipeline = [
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
    window_size=-1,
    save_dir='results/k_means_ac_preprocess',
    experiment_name='KMeans_ac_preprocess_Accuracy',
    column_names=["Axis1_A", "Axis2_A", "Axis3_A", "Axis1_N", "Axis2_N", "Axis3_N"]

)
