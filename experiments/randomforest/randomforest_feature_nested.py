from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from digital_health_project.models.models import AlignedTimeSeriesKMeans
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.features import HandFeatureExtractor


TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


rf_pipeline = [
    ('feature_extractor', HandFeatureExtractor()),
    ('rf', RandomForestClassifier())
]

rf_params = [
    {   
        "feature_extractor__mode": [
        "active_only",
        "active_mirror",
        "bilateral_only",
        "all_features"
        ],

        'rf__n_estimators': [100, 200, 500],
        'rf__max_depth': [None, 5, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__class_weight': [None, 'balanced']
    }
]

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=rf_pipeline,
    param_grid=rf_params,
    save_dir='results/rf_feature_nested',
    experiment_name='RF_feature_nested_Accuracy'
)