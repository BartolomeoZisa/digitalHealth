from sklearn.pipeline import Pipeline
from sktime.classification.dictionary_based import BOSSEnsemble

from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.difference_assimmetry import (
    InterHandProcessor,
    SktimeFormatTransformer
)

TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'

boss_pipeline = [
    ('inter_hand', InterHandProcessor()),
    ('sktime_formatter', SktimeFormatTransformer()),
    ('clf', BOSSEnsemble())  # Direct sktime classifier
]

boss_params = [{
    'inter_hand__mode': ['diff', 'asymmetry_index'],
    'clf__feature_selection': ["chi2", "none"]
}]

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=boss_pipeline,
    param_grid=boss_params,
    experiment_name='BOSS_Ensemble_Classifier',
    save_dir='results/boss_ac_preprocessed',
)