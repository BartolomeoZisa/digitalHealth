from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch import nn
import torch
from digital_health_project.features.stft import STFTTransformer
from digital_health_project.models.cnn import MultiBranchCNN
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.models.cnn import EEGNet
from sklearn.model_selection import GroupShuffleSplit


class FloatBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        return super().forward(input, target.float())

TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'

# Configure the skorch wrapper for EEGNet
net = NeuralNetClassifier(
    module=EEGNet,
    module__input_channels=6,
    module__output_dim=1,
    module__F1=8,           # Start small for few data
    module__D=2,
    module__dropout=0.5,    # High dropout is vital for small datasets
    criterion=FloatBCEWithLogitsLoss,
    optimizer=torch.optim.AdamW, 
    optimizer__weight_decay=0.05, # Aggressive weight decay
    lr=0.001,
    max_epochs=1000,
    batch_size=32,
    # Internal validation split using Groups to prevent leakage
    train_split=ValidSplit(cv=GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)),
    callbacks=[EarlyStopping(patience=20, load_best=True)],
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# -------------------------------------------------------------
# 3. Run Pipeline
# -------------------------------------------------------------

pipeline_steps = [
    ('eegnet', net)
]

param_grid = {
    # F1=4 and D=2 results in only a few thousand parameters
    'eegnet__module__F1': [4, 8],
    'eegnet__module__D': [2],
    'eegnet__lr': [0.001, 0.0001],
    'eegnet__optimizer__weight_decay': [1e-2, 1e-4]
}

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=pipeline_steps,
    param_grid=param_grid,
    experiment_name='eegnet_low_params',
    n_jobs=1 
)