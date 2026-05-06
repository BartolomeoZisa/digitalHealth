from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch import nn
import torch
from digital_health_project.features.stft import STFTTransformer
from digital_health_project.models.cnn import MultiBranchCNN
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.models.cnn import TimeSeriesCNN
from sklearn.model_selection import GroupShuffleSplit

TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'


# 1. Define the skorch wrapper
# ValidSplit(0.1) ensures early stopping uses 10% of the CURRENT training fold
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping

# Create a small wrapper to cast Y to float during loss calculation
class FloatBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        return super().forward(input, target.float())

# Update your skorch wrapper definition
# 1. Update the skorch wrapper to use AdamW
TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'

# Update the skorch wrapper to use GroupShuffleSplit
net = NeuralNetBinaryClassifier(
    module=TimeSeriesCNN,
    module__input_channels=6,
    module__output_dim=1,
    criterion=FloatBCEWithLogitsLoss,
    optimizer=torch.optim.AdamW, 
    optimizer__weight_decay=0.01,
    lr=0.001,
    max_epochs=1000,
    batch_size=32,
    
    # -------------------------------------------------------------
    # Use GroupShuffleSplit to avoid window leakage during early stopping!
    train_split=ValidSplit(cv=GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)),
    # -------------------------------------------------------------
    
    callbacks=[EarlyStopping(patience=20)],
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

pipeline_steps = [
    ('cnn', net)
]

param_grid = {
    'cnn__module__num_filters': [16, 32],
    'cnn__module__kernel_size': [3, 5],
    'cnn__lr': [0.0001, 0.001],
    'cnn__optimizer__weight_decay': [1e-2, 1e-4]
}

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=pipeline_steps,
    param_grid=param_grid,
    experiment_name='1dcnn',
    n_jobs=1 
)
