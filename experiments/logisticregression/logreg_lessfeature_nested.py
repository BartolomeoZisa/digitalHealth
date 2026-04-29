from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from digital_health_project.models.models import AlignedTimeSeriesKMeans
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.features import DualHandFeatureExtractor


TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


logreg_pipeline = [
    ('feature_extractor', DualHandFeatureExtractor()),
    ('scaler', StandardScaler()),   
    ('logreg', LogisticRegression(max_iter=1000))
]

logreg_params = [
    {
        "feature_extractor__mode": [
            "base",
            "preprocessed"
        ],

        'logreg__penalty': ['l2'],
        'logreg__C': [0.01, 0.1, 1, 10, 100],
        'logreg__solver': ['lbfgs'],
    },
    {   
        "feature_extractor__mode": [
        "active_only",
        "active_mirror",
        "bilateral_only",
        "all_features"
        ],

        'logreg__penalty': ['l1'],
        'logreg__C': [0.01, 0.1, 1, 10, 100],
        'logreg__solver': ['liblinear'],
    }
]

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=logreg_pipeline,
    param_grid=logreg_params,
    save_dir='results/logreg_feature_nested',
    experiment_name='LogReg_feature_nested_Accuracy'
)