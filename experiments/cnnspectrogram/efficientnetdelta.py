from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch import nn
import torch
from digital_health_project.features.stft import STFTTransformer
from digital_health_project.models.cnn import MultiBranchCNN
from digital_health_project.utils.pipeline import run_classification_pipeline

# 1. Define the skorch wrapper
# ValidSplit(0.2) ensures early stopping uses 20% of the CURRENT training fold
net = NeuralNetClassifier(
    module=MultiBranchCNN,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    max_epochs=50,
    batch_size=32,
    train_split=ValidSplit(0.1), 
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=[
        EarlyStopping(monitor='valid_loss', patience=5, restore_best_weights=True)
    ],
)

# 2. Define Pipeline Steps
pipeline_steps = [
    ('stft', STFTTransformer(fs=80, nperseg=240, noverlap=120)),
    ('cnn', net)
]

# 3. Define Parameter Grid
# You can toggle between architectures from  here
param_grid = {
    'cnn__module__architecture': ['C1', 'C2'],
    'cnn__lr': [0.0001]
}

run_classification_pipeline(
    td_path='path_to_td.csv',
    ucp_path='path_to_ucp.csv',
    pipeline_steps=pipeline_steps,
    param_grid=param_grid,
    window_size=-1, # Set to -1 to use full window size (no sliding)
    experiment_name='efficientnetdelta',
    n_jobs=1 # Set to 1 when using GPU/Neural Networks to avoid memory errors
)