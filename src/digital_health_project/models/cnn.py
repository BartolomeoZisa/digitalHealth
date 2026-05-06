import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
from torchvision import models

class MultiBranchCNN(nn.Module):
    def __init__(self, architecture='C1', num_classes=2, dropout_rate=0.5, pretrained=True):
        super(MultiBranchCNN, self).__init__()
        self.architecture = architecture
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Determine input channels based on architecture
        base_in_channels = 2 if architecture == 'C3' else 1
        
        # Base model selection
        if architecture in ['C1', 'C3']:
            # EfficientNet-B0
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.base = models.efficientnet_b0(weights=weights)
            
            # Update First Layer
            old_conv = self.base.features[0][0]
            self.base.features[0][0] = self._get_new_conv(old_conv, base_in_channels)
            
            # Setup Classifier
            self.feature_dim = self.base.classifier[1].in_features
            self.base.classifier = nn.Identity() 
        else:
            # ResNet18
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.base = models.resnet18(weights=weights)
            
            # Update First Layer
            old_conv = self.base.conv1
            self.base.conv1 = self._get_new_conv(old_conv, base_in_channels)
            
            # Setup Classifier
            self.feature_dim = self.base.fc.in_features
            self.base.fc = nn.Identity()

        # Final classification head
        if architecture in ['C1', 'C2']:
            self.classifier = nn.Linear(self.feature_dim * 2, num_classes)
        else:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _get_new_conv(self, old_conv, in_channels):
        """
        Adapts a 3-channel pretrained conv layer to accept 'in_channels'.
        """
        new_conv = nn.Conv2d(
            in_channels, 
            old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=old_conv.bias is not None
        )
        
        # Optional: Copy weights from the first in_channels of the pretrained model
        # This is better than random initialization for 1 or 2 channel inputs
        with torch.no_grad():
            if old_conv.in_channels >= in_channels:
                new_conv.weight.copy_(old_conv.weight[:, :in_channels, :, :])
            else:
                # If for some reason the old conv had fewer channels (rare for pretrained)
                new_conv.weight[:, :old_conv.in_channels, :, :].copy_(old_conv.weight)
        
        return new_conv

    def forward(self, x):
        x = x.float() 
        
        if self.architecture in ['C1', 'C2']:
            # Shared weight branches
            branch1 = self.base(x[:, 0:1, :, :])
            branch2 = self.base(x[:, 1:2, :, :])
            merged = torch.cat((branch1, branch2), dim=1)
            merged = self.dropout(merged)
            return self.classifier(merged)
            
        elif self.architecture == 'C3':
            spec = x[:, 0:1, :, :]
            delta = torch.zeros_like(spec)
            delta[:, :, :, 1:] = spec[:, :, :, 1:] - spec[:, :, :, :-1]
            
            combined = torch.cat((spec, delta), dim=1)
            features = self.base(combined)
            features = self.dropout(features)
            return self.classifier(features)            

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesCNN(nn.Module):
    def __init__(
        self, 
        input_channels=6, 
        num_filters=32, 
        kernel_size=3, 
        pool_size=2, 
        dropout=0.2, 
        output_dim=1
    ):
        super(TimeSeriesCNN, self).__init__()
        
        # Layer 1: Feature Extraction
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        # Layer 2: Deeper Features
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        
        self.pool = nn.AdaptiveMaxPool1d(1) # Global Pooling handles variable window lengths
        self.dropout = nn.Dropout(dropout)
        
        # Final classification layer
        # Using output_dim=1 for Binary Classification (BCEWithLogitsLoss)
        self.fc = nn.Linear(num_filters * 2, output_dim)

    def forward(self, x):
        # Ensure input is float32 for the Conv layers
        x = x.float() 
        
        # x shape: (batch, window_size, channels) -> (batch, channels, window_size)
        x = x.transpose(1, 2) 
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.pool(x).squeeze(-1) 
        x = self.dropout(x)
        
        # Return logits (BCEWithLogitsLoss handles the sigmoid internally)
        return self.fc(x).squeeze(-1)



