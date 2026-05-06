from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch import nn
import torch
from digital_health_project.models.cnn import MultiBranchCNN
from digital_health_project.utils.pipeline import run_classification_pipeline
from digital_health_project.models.cnn import FCN
from sklearn.model_selection import GroupShuffleSplit


class FloatBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        return super().forward(input, target.float())

TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'

net = NeuralNetClassifier(
    module=FCN,
    module__filter_scale=1.0,
    module__input_channels=6,
    module__output_dim=1,
    criterion=FloatBCEWithLogitsLoss,
    optimizer=torch.optim.AdamW, 
    optimizer__weight_decay=0.01,
    lr=0.001,
    max_epochs=1000,
    batch_size=32,
    train_split=ValidSplit(cv=GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)),
    callbacks=[EarlyStopping(patience=20)],
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# -------------------------------------------------------------
# 3. Pipeline and Grid Search
# -------------------------------------------------------------
pipeline_steps = [
    ('fcn', net)
]

param_grid = {
            # Try different model sizes: 0.25 (small), 0.5 (medium), 1.0 (standard literature)
    'fcn__lr': [0.0001, 0.001],
    'fcn__optimizer__weight_decay': [1e-2, 1e-4],
    'fcn__module__filter_scale': [0.25, 0.5, 1.0]
}

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=pipeline_steps,
    param_grid=param_grid,
    experiment_name='fcn_experiment',
    n_jobs=1 
)