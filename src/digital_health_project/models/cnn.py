import torch
import torch.nn as nn
import torchvision.models as models

class MultiBranchCNN(nn.Module):
    def __init__(self, architecture='C1', num_classes=2, dropout_rate=0.5):
        super(MultiBranchCNN, self).__init__()
        self.architecture = architecture
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # LOGIC FIX: 
        # C1 and C2 slice the input into 1-channel branches.
        # C3 stacks the input into a 2-channel (spec+delta) image.
        base_in_channels = 2 if architecture == 'C3' else 1
        
        # Base model selection
        if architecture in ['C1', 'C3']:
            # EfficientNet-B0
            self.base = models.efficientnet_b0(weights=None)
            # Use base_in_channels instead of in_channels here
            self.base.features[0][0] = nn.Conv2d(base_in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.feature_dim = self.base.classifier[1].in_features
            self.base.classifier = nn.Identity() 
        else:
            # ResNet18
            self.base = models.resnet18(weights=None)
            # Use base_in_channels instead of in_channels here
            self.base.conv1 = nn.Conv2d(base_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_dim = self.base.fc.in_features
            self.base.fc = nn.Identity()

        # Final classification head (unchanged)
        if architecture in ['C1', 'C2']:
            self.classifier = nn.Linear(self.feature_dim * 2, num_classes)
        else:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # Force input to float32 to prevent DoubleTensor errors
        x = x.float() 
        
        # Input x shape: (batch, channels, freq, time)
        if self.architecture in ['C1', 'C2']:
            # Two-branch shared weights
            branch1 = self.base(x[:, 0:1, :, :])
            branch2 = self.base(x[:, 1:2, :, :])
            merged = torch.cat((branch1, branch2), dim=1)
            merged = self.dropout(merged)
            return self.classifier(merged)
            
        elif self.architecture == 'C3':
            spec = x[:, 0:1, :, :]
            delta = torch.zeros_like(spec) # This will now be float32
            delta[:, :, :, 1:] = spec[:, :, :, 1:] - spec[:, :, :, :-1]
            
            combined = torch.cat((spec, delta), dim=1)
            features = self.base(combined)
            features = self.dropout(features)
            return self.classifier(features)
            