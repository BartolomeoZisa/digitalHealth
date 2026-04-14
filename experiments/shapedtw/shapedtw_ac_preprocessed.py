from sklearn.pipeline import Pipeline
from digital_health_project.models.models import AlignedTimeSeriesKMeans, ShapeDTWClassifier
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.features.difference_assimmetry import InterHandProcessor, SktimeFormatTransformer


TD_PATH = 'data/td/bbt_td_1sec_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_1sec_anon.csv'


shapedtw_pipeline = [
        ('inter_hand', InterHandProcessor()),
        ('sktime_formatter', SktimeFormatTransformer()),
        ('clf', ShapeDTWClassifier()) # Using the new wrapper
    ]

shapedtw_params = [{
        'inter_hand__mode': ['diff', 'asymmetry_index'],
        'clf__shape_descriptor_function': ['raw', 'paa'],
    }]
    

run_classification_pipeline(
        td_path=TD_PATH, ucp_path=UCP_PATH,
        pipeline_steps=shapedtw_pipeline,
        param_grid=shapedtw_params,
        experiment_name='ShapeDTW_Direct_Classifier',
        save_dir='results/shapedtw_ac_preprocessed',
        window_size=-1,  # No windowing, use full time series
        column_names=["Axis1_A", "Axis2_A", "Axis3_A", "Axis1_N", "Axis2_N", "Axis3_N"]
    )