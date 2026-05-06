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

    def forward(self, x, **kwargs):
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

    def forward(self, x, **kwargs):
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


import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super(InceptionBlock, self).__init__()
        
        # 1. Bottleneck Layer (reduces computation)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        
        # 2. Parallel Convolution branches
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, out_channels, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        
        # 3. Max Pooling branch
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        
        # 4. Final normalization and activation
        self.batch_norm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # The bottleneck is applied only to the convolution branches
        bottlenecked = self.bottleneck(x)
        
        # Convolve parallel branches
        conv_out = [layer(bottlenecked) for layer in self.conv_layers]
        
        # Maxpool branch uses the raw input x
        pool_out = [self.maxpool_branch(x)]
        
        # Concatenate all 4 branches along the channel dimension
        out = torch.cat(conv_out + pool_out, dim=1)
        return self.relu(self.batch_norm(out))


class InceptionTime(nn.Module):
    def __init__(
        self,
        input_channels=6,
        num_blocks=6,
        num_channels=32,
        kernel_sizes=[9, 19, 39],
        dropout=0.1,
        output_dim=1,
    ):
        super(InceptionTime, self).__init__()
        
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        
        # Each block outputs (3 conv branches + 1 maxpool branch) * num_channels
        self.block_out_channels = num_channels * (len(kernel_sizes) + 1)
        
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        
        for i in range(num_blocks):
            in_ch = input_channels if i == 0 else self.block_out_channels
            
            # Add the Inception Block
            self.blocks.append(
                InceptionBlock(in_ch, num_channels, kernel_sizes)
            )
            
            # Add Residual shortcut every 3 blocks
            if i % 3 == 2:
                shortcut_in_ch = input_channels if i == 2 else self.block_out_channels
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(shortcut_in_ch, self.block_out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.block_out_channels)
                ))

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.block_out_channels, output_dim)
        
    def forward(self, x, **kwargs):
        x = x.float()
        
        # Handle Skorch/Standard input shape: (batch, time, channels) -> (batch, channels, time)
        if x.dim() == 3 and x.shape[1] != self.input_channels:
            x = x.transpose(1, 2)
            
        res_input = x # The input for the first residual shortcut
        
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
            
            # Apply residual addition every 3 blocks
            if i % 3 == 2:
                shortcut_layer = self.shortcuts[i // 3]
                x = x + shortcut_layer(res_input)
                x = F.relu(x)
                res_input = x # Update res_input for the next trio
                
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)



import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_channels=6, output_dim=1, filter_scale=1.0):
        super(FCN, self).__init__()
        
        # Calculate scaled filters
        f1 = int(128 * filter_scale)
        f2 = int(256 * filter_scale)
        f3 = int(128 * filter_scale)  # This is the output dimension before GAP

        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, f1, kernel_size=8, padding=4),
            nn.BatchNorm1d(f1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(f1, f2, kernel_size=5, padding=2),
            nn.BatchNorm1d(f2),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(f2, f3, kernel_size=3, padding=1),
            nn.BatchNorm1d(f3),
            nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # FIX: The input to the linear layer must be f3 (the scaled size)
        self.classifier = nn.Linear(f3, output_dim) 

    def forward(self, x, **kwargs):
        x = x.float()
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.gap(x).squeeze(-1) # Results in shape (batch, f3)
        return self.classifier(x).squeeze(-1)



import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, input_channels=6, output_dim=1, kernel_length=64, F1=8, D=2, dropout=0.5):
        super(EEGNet, self).__init__()
        
        # F1: Number of temporal filters
        # D: Depth multiplier (spatial filters)
        F2 = F1 * D  # Number of pointwise filters
        
        # Block 1: Temporal & Spatial Convolution
        self.block1 = nn.Sequential(
            # Temporal Conv
            nn.Conv1d(input_channels, F1, kernel_size=kernel_length, padding=kernel_length // 2, bias=False),
            nn.BatchNorm1d(F1),
            # Depthwise Conv (Spatial Filter)
            # groups=F1 forces the model to learn spatial patterns per temporal filter
            nn.Conv1d(F1, F2, kernel_size=1, groups=F1, bias=False),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(dropout)
        )
        
        # Block 2: Separable Convolution
        # This is a Depthwise Conv followed by a Pointwise (1x1) Conv
        self.block2 = nn.Sequential(
            nn.Conv1d(F2, F2, kernel_size=16, padding=8, groups=F2, bias=False),
            nn.Conv1d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.AvgPool1d(8),
            nn.Dropout(dropout)
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(F2, output_dim)

    def forward(self, x, **kwargs):
        x = x.float()
        
        # Ensure (batch, channels, time)
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)
            
        x = self.block1(x)
        x = self.block2(x)
        
        x = self.gap(x).squeeze(-1)
        return self.classifier(x).squeeze(-1)