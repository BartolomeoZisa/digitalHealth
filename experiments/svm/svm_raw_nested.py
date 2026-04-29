from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from digital_health_project.utils.pipeline import run_classification_pipeline


TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


flatten = FunctionTransformer(lambda X: X.reshape(len(X), -1))

svm_pipeline = [
    ('flatten', flatten),
    ('scaler', StandardScaler()),
    ('svm', SVC())
]

svm_params = [
    {
        'svm__kernel': ['rbf'],
        'svm__C': [0.01, 0.1, 1, 10, 100],
        'svm__gamma': [0.001, 0.01, 0.1, 1],
    }
]

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=svm_pipeline,
    param_grid=svm_params,
    save_dir='results/svm_flatten_nested',
    experiment_name='SVM_flatten_nested'
)