from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from digital_health_project.models.models import AlignedTimeSeriesKMeans
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.features import HandFeatureExtractor


TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'

    
svm_pipeline = [
    ('feature_extractor', HandFeatureExtractor()),
    ('scaler', StandardScaler()),
    ('svm', SVC())
]

svm_params = [
    {   
        "feature_extractor__mode": [
        "active_only",
        "active_mirror",
        "bilateral_only",
        "all_features"
        ],

        'svm__kernel': ['rbf'],
        'svm__C': [0.01, 0.1, 1, 10, 100],
        'svm__gamma': [0.001, 0.01, 0.1, 1],
    }
]

run_classification_pipeline(
    td_path=TD_PATH, ucp_path=UCP_PATH,
    pipeline_steps=svm_pipeline,
    param_grid=svm_params,
    save_dir='results/svm_feature_nested',
    experiment_name='SVM_feature_nested_Accuracy'
)
