import torch
import torch.nn as nn
import torchvision.models as models

class MultiBranchCNN(nn.Module):
    def __init__(self, architecture='C1', num_classes=2, in_channels=1):
        super(MultiBranchCNN, self).__init__()
        self.architecture = architecture
        
        # Base model selection
        if architecture in ['C1', 'C3']:
            # EfficientNet-B0 (C1: shared weights, C3: delta)
            self.base = models.efficientnet_b0(weights=None)
            self.base.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.feature_dim = self.base.classifier[1].in_features
            self.base.classifier = nn.Identity() 
        else:
            # ResNet18 (C2: shared weights)
            self.base = models.resnet18(weights=None)
            self.base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_dim = self.base.fc.in_features
            self.base.fc = nn.Identity()

        # Final classification head
        if architecture in ['C1', 'C2']:
            self.classifier = nn.Linear(self.feature_dim * 2, num_classes)
        else:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # Input x shape: (batch, channels, freq, time)
        
        if self.architecture in ['C1', 'C2']:
            # Two-branch shared weights: Process channel 0 and 1 separately
            branch1 = self.base(x[:, 0:1, :, :])
            branch2 = self.base(x[:, 1:2, :, :])
            merged = torch.cat((branch1, branch2), dim=1)
            return self.classifier(merged)
            
        elif self.architecture == 'C3':
            # Delta Spectrogram: channel 0 + its time-derivative
            spec = x[:, 0:1, :, :]
            delta = torch.zeros_like(spec)
            delta[:, :, :, 1:] = spec[:, :, :, 1:] - spec[:, :, :, :-1]
            
            # Combine to (batch, 2, freq, time) 
            # If base model was adjusted for 2 channels, we feed it directly
            combined = torch.cat((spec, delta), dim=1)
            # Note: For C3, ensure in_channels=2 was passed to __init__
            features = self.base(combined)
            return self.classifier(features)