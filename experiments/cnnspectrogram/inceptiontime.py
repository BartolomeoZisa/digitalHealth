from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch import nn
import torch
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.models.cnn import InceptionTime
from sklearn.model_selection import GroupShuffleSplit

TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'

# Wrapper to cast Y to float during loss calculation
class FloatBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        return super().forward(input, target.float())

# InceptionTime (Fawaz et al., 2020)
# Multi-scale 1D convolutions capture patterns at different temporal granularities
net = NeuralNetBinaryClassifier(
    module=InceptionTime,
    module__input_channels=6,
    module__output_dim=1,
    criterion=FloatBCEWithLogitsLoss,
    optimizer=torch.optim.AdamW,
    optimizer__weight_decay=0.01,
    lr=0.001,
    max_epochs=1000,
    batch_size=32,

    # GroupShuffleSplit to avoid window leakage during early stopping
    train_split=ValidSplit(cv=GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)),

    callbacks=[EarlyStopping(patience=10)],
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

pipeline_steps = [
    ('cnn', net)
]

param_grid = {
    'cnn__module__num_blocks': [6],
    'cnn__module__num_channels': [32, 64],
    'cnn__lr': [0.00001, 0.0001],
    'cnn__optimizer__weight_decay': [1e-4, 1e-2],
}

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=pipeline_steps,
    param_grid=param_grid,
    experiment_name='inceptiontime',
    n_jobs=1,
)
