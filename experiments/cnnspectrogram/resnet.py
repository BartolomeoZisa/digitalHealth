from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch import nn
import torch
from digital_health_project.features.stft import STFTTransformer
from digital_health_project.models.cnn import MultiBranchCNN
from digital_health_project.utils.pipeline import run_classification_pipeline

TD_PATH = 'data/td/bbt_td_raw_anon.csv'
UCP_PATH = 'data/ucp/bbt_ucp_raw_anon.csv'

# 1. Define the skorch wrapper
# ValidSplit(0.1) ensures early stopping uses 10% of the CURRENT training fold
net = NeuralNetClassifier(
    module=MultiBranchCNN,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,     # Switched to AdamW
    optimizer__weight_decay=0.01,    # Added L2 regularization
    max_epochs=1000,
    batch_size=32,
    train_split=ValidSplit(0.1), 
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=[
        # Lowered patience slightly; 20 is fine, but 10-15 catches overfitting faster
        EarlyStopping(monitor='valid_loss', patience=20) 
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
    'cnn__module__architecture': ['C2'],
    'cnn__module__dropout_rate': [0.3, 0.5],
    'cnn__optimizer__weight_decay': [1e-4, 1e-2],
    'cnn__lr': [0.0001, 0.001]
}

run_classification_pipeline(
    td_path=TD_PATH,
    ucp_path=UCP_PATH,
    pipeline_steps=pipeline_steps,
    param_grid=param_grid,
    window_size=-1, # Set to -1 to use full window size (no sliding)
    experiment_name='resnet',
    n_jobs=1 # Set to 1 when using GPU/Neural Networks to avoid memory errors
)