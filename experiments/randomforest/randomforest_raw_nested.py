from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from digital_health_project.utils.pipeline import run_classification_pipeline


TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


flatten = FunctionTransformer(lambda X: X.reshape(len(X), -1))

rf_pipeline = [
    ('flatten', flatten),
    ('rf', RandomForestClassifier())
]

rf_params = [
    {
        'rf__n_estimators': [200, 500],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2],
        'rf__max_features': ['sqrt', 'log2'],
    }
]

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=rf_pipeline,
    param_grid=rf_params,
    save_dir='results/rf_flatten_nested',
    experiment_name='RF_flatten_nested'
)