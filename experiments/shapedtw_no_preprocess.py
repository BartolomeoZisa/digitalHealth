from sklearn.pipeline import Pipeline
from digital_health_project.models.models import ShapeDTWClassifier
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.difference_assimmetry import InterHandProcessor, SktimeFormatTransformer, PairedSignalWindower


TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


shapedtw_pipeline = [
        ('sktime_formatter', SktimeFormatTransformer()),
        ('clf', ShapeDTWClassifier()) # Using the new wrapper
    ]

shapedtw_params = [{
        'clf__shape_descriptor_function': ['raw', 'paa'],
    }]
    

run_classification_pipeline(
        td_path=TD_PATH, ucp_path=UCP_PATH,
        pipeline_steps=shapedtw_pipeline,
        param_grid=shapedtw_params,
        experiment_name='ShapeDTW_Direct_Classifier',
    )