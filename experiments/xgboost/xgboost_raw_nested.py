from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from digital_health_project.utils.pipeline import run_classification_pipeline


TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


flatten = FunctionTransformer(lambda X: X.reshape(len(X), -1))

xgb_pipeline = [
    ('flatten', flatten),
    ('xgb', XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    ))
]

xgb_params = [
    {
        'xgb__n_estimators': [200, 500],
        'xgb__max_depth': [5, 7],
        'xgb__learning_rate': [0.05, 0.1],
        'xgb__subsample': [0.8],
        'xgb__colsample_bytree': [0.8],
    }
]

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=xgb_pipeline,
    param_grid=xgb_params,
    save_dir='results/xgb_flatten_nested',
    experiment_name='XGB_flatten_nested'
)