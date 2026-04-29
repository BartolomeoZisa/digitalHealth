from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from digital_health_project.utils.pipeline import run_classification_pipeline


TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


flatten = FunctionTransformer(lambda X: X.reshape(len(X), -1))

logreg_pipeline = [
    ('flatten', flatten),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
]

logreg_params = [
    {
        'logreg__penalty': ['l2'],
        'logreg__C': [0.01, 0.1, 1, 10, 100],
        'logreg__solver': ['lbfgs'],
    },
    {
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
    save_dir='results/logreg_flatten_nested',
    experiment_name='LogReg_flatten_nested'
)