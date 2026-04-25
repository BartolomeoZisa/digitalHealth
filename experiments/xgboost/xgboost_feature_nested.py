from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from digital_health_project.models.models import AlignedTimeSeriesKMeans
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.features import HandFeatureExtractor


TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


xgb_pipeline = [
    ('feature_extractor', HandFeatureExtractor()),
    ('xgb', XGBClassifier(
        eval_metric='logloss',
        random_state=42
    ))
]

xgb_params = [
    {   
        "feature_extractor__mode": [
        "active_only",
        "active_mirror",
        "bilateral_only",
        "all_features"
        ],
        'xgb__n_estimators': [100, 200, 500],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0],
        'xgb__gamma': [0, 0.1, 0.5],
        'xgb__reg_alpha': [0, 0.1, 1],
        'xgb__reg_lambda': [1, 5, 10],
        'xgb__scale_pos_weight': [1, 2, 5]
    }
]

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=xgb_pipeline,
    param_grid=xgb_params,
    window_size=-1,
    save_dir='results/xgb_feature_nested',
    experiment_name='XGB_feature_nested_Accuracy'
)