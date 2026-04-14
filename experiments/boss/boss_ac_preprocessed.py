from sklearn.pipeline import Pipeline
from sktime.classification.dictionary_based import BOSSEnsemble

from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.difference_assimmetry import (
    InterHandProcessor,
    SktimeFormatTransformer
)

TD_PATH = 'data/td/bbt_td_1sec_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_1sec_anon.csv'

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
    window_size=-1,  # No windowing
    column_names=["Axis1_A", "Axis2_A", "Axis3_A", "Axis1_N", "Axis2_N", "Axis3_N"]
)