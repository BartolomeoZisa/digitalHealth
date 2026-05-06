from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch import nn
import torch
from digital_health_project.features.stft import STFTTransformer
from digital_health_project.models.cnn import MultiBranchCNN
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.models.cnn import TimeSeriesCNN

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
from skorch.dataset import GroupedValidSplit
from torch.optim import AdamW

# 1. Define the skorch wrapper
# Using GroupedValidSplit requires passing 'groups' to the .fit() method later.
# In skorch, this is handled by specifying the split and ensuring the 
# underlying pipeline or fit call provides the 'groups' metadata.

net = NeuralNetBinaryClassifier(
    module=TimeSeriesCNN,
    module__input_channels=6,
    module__output_dim=1,
    criterion=FloatBCEWithLogitsLoss, 
    optimizer=AdamW,                  # Swapped to AdamW
    optimizer__weight_decay=0.01,    # Common default for AdamW
    lr=0.001,
    max_epochs=50,
    batch_size=32,
    # Use GroupedValidSplit: 10% of groups will be used for validation
    train_split=GroupedValidSplit(0.1), 
    callbacks=[EarlyStopping(patience=10)],
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# 2. Define Pipeline Steps
pipeline_steps = [
    ('cnn', net)
]


# 2. Define Pipeline Steps
pipeline_steps = [
    ('cnn', net)
]

# 3. Define Parameter Grid
# You can toggle between architectures from  here
param_grid = {
    'cnn__module__num_filters': [16, 32],
    'cnn__module__kernel_size': [3, 5],
    'cnn__optimizer__weight_decay': [1e-4, 1e-2], # Added weight decay to grid
    'cnn__lr': [0.01, 0.001]
}


run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=pipeline_steps,
    param_grid=param_grid,
    #window_size=-1, # Set to -1 to use full window size (no sliding)
    experiment_name='1dcnn',
    n_jobs=1 # Set to 1 when using GPU/Neural Networks to avoid memory errors
)