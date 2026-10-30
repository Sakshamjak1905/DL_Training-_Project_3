"""
=============================================================================
CNN Model Architecture & Deep Learning Toolkit
Course / Project: Deep Learning & Computer Vision
Author: CNN Image Classifier Studio
Description: Modular Convolutional Neural Network with BatchNorm, Dropout,
             Grad-CAM Explainable AI (XAI), and Feature Map Visualizers.
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Core CNN Architecture
# -----------------------------------------------------------------------------
class ConvNet(nn.Module):
    """
    Modular Convolutional Neural Network.
    
    Architecture:
      - Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2)
      - Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2)
      - Block 3 (Optional): Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2)
      - Classifier: AdaptiveAvgPool2d((4, 4)) -> Flatten -> Dense(128) -> ReLU -> Dropout -> Dense(num_classes)
    """
    def __init__(self, in_channels=3, num_classes=10, dropout=0.3, num_blocks=2):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Optional Block 3
        if num_blocks >= 3:
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(2, 2)
            final_channels = 128
        else:
            self.conv3 = None
            final_channels = 64
            
        # Adaptive pooling ensures consistent dimension regardless of image resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(final_channels * 4 * 4, 128)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Hook holders for Grad-CAM
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3 (if enabled)
        if self.conv3 is not None:
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)
            
        # Register hook on the last convolutional activation for Grad-CAM
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x
        
        # Classifier Head
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def get_last_conv_layer(self):
        """Returns the last convolutional layer in the feature extractor."""
        if self.conv3 is not None:
            return self.conv3
        return self.conv2


# -----------------------------------------------------------------------------
# 2. Builder Function (Backwards Compatible with build_cnn)
# -----------------------------------------------------------------------------
def build_cnn(input_shape=(32, 32, 3), num_classes=10, dropout=0.3, num_blocks=2):
    """
    Instantiates and returns the ConvNet model.
    Accepts input_shape either as (H, W, C) or (C, H, W).
    """
    if len(input_shape) == 3:
        if input_shape[0] in [1, 3]:
            in_channels = input_shape[0]
        else:
            in_channels = input_shape[2]
    else:
        in_channels = 3
        
    model = ConvNet(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
        num_blocks=num_blocks
    )
    return model


# -----------------------------------------------------------------------------
# 3. Model Information & Parameter Counting
# -----------------------------------------------------------------------------
def count_parameters(model):
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_layer_breakdown(model):
    """Returns a list of dicts describing layer details for academic display."""
    layers_info = []
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        layers_info.append({
            "Layer Name": name,
            "Type": module.__class__.__name__,
            "Parameters": f"{param_count:,}",
            "Trainable": any(p.requires_grad for p in module.parameters())
        })
    return layers_info


# -----------------------------------------------------------------------------
# 4. Explainable AI: Grad-CAM Implementation
# -----------------------------------------------------------------------------
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes where the neural network is 'looking' when making a decision.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        input_tensor: (1, C, H, W) normalized torch Tensor
        target_class: integer class index (defaults to argmax prediction)
        returns: (H, W) numpy float array in range [0, 1]
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        # Backward pass to get gradients
        score.backward()
        
        # Pull gradients and activations from model hooks
        gradients = self.model.gradients
        activations = self.model.activations
        
        if gradients is None or activations is None:
            # Fallback if hooks were not triggered
            return np.ones((input_tensor.shape[2], input_tensor.shape[3])) * 0.5
            
        # Global Average Pooling of gradients (weights alpha_k)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        # Weighted sum of feature maps
        cam = torch.sum(weights * activations, dim=1).squeeze()
        # ReLU to keep only features that have a positive influence
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        
        # Normalize to [0, 1]
        if np.max(cam) != np.min(cam):
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        else:
            cam = np.zeros_like(cam)
            
        # Resize to input resolution
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = cam_img.resize((input_tensor.shape[3], input_tensor.shape[2]), Image.BILINEAR)
        return np.array(cam_resized) / 255.0


def overlay_heatmap(orig_img, heatmap, alpha=0.45, colormap="jet"):
    """
    Superimposes the Grad-CAM heatmap over the original PIL image.
    orig_img: PIL Image
    heatmap: 2D numpy array [0, 1]
    returns: PIL Image
    """
    import matplotlib.cm as cm
    
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(orig_img.size, Image.BILINEAR)
    heatmap_arr = np.array(heatmap_resized) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_arr)[:, :, :3]  # drop alpha
    colored_heatmap = np.uint8(colored_heatmap * 255)
    
    orig_arr = np.array(orig_img.convert("RGB"))
    blended = np.uint8(orig_arr * (1 - alpha) + colored_heatmap * alpha)
    return Image.fromarray(blended)


# -----------------------------------------------------------------------------
# 5. Intermediate Feature Map Activations
# -----------------------------------------------------------------------------
def get_feature_maps(model, input_tensor):
    """
    Passes an input tensor through the model and captures activation maps
    from conv1, conv2, and conv3 (if present).
    returns: dict of {layer_name: numpy array (channels, H, W)}
    """
    model.eval()
    feature_maps = {}
    with torch.no_grad():
        x = input_tensor
        x = model.conv1(x)
        x = model.bn1(x)
        x = F.relu(x)
        feature_maps["Conv Layer 1 (Low-level Edges/Textures)"] = x.squeeze(0).cpu().numpy()
        x = model.pool1(x)
        
        x = model.conv2(x)
        x = model.bn2(x)
        x = F.relu(x)
        feature_maps["Conv Layer 2 (Mid-level Shapes/Patterns)"] = x.squeeze(0).cpu().numpy()
        x = model.pool2(x)
        
        if model.conv3 is not None:
            x = model.conv3(x)
            x = model.bn3(x)
            x = F.relu(x)
            feature_maps["Conv Layer 3 (High-level Semantic Parts)"] = x.squeeze(0).cpu().numpy()
            
    return feature_maps



# -----------------------------------------------------------------------------
# 3. Model Information & Parameter Counting
# -----------------------------------------------------------------------------
def count_parameters(model):
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_layer_breakdown(model):
    """Returns a list of dicts describing layer details for academic display."""
    layers_info = []
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        layers_info.append({
            "Layer Name": name,
            "Type": module.__class__.__name__,
            "Parameters": f"{param_count:,}",
            "Trainable": any(p.requires_grad for p in module.parameters())
        })
    return layers_info


# -----------------------------------------------------------------------------
# 4. Explainable AI: Grad-CAM Implementation
# -----------------------------------------------------------------------------
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes where the neural network is 'looking' when making a decision.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        input_tensor: (1, C, H, W) normalized torch Tensor
        target_class: integer class index (defaults to argmax prediction)
        returns: (H, W) numpy float array in range [0, 1]
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        # Backward pass to get gradients
        score.backward()
        
        # Pull gradients and activations from model hooks
        gradients = self.model.gradients
        activations = self.model.activations
        
        if gradients is None or activations is None:
            # Fallback if hooks were not triggered
            return np.ones((input_tensor.shape[2], input_tensor.shape[3])) * 0.5
            
        # Global Average Pooling of gradients (weights alpha_k)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        # Weighted sum of feature maps
        cam = torch.sum(weights * activations, dim=1).squeeze()
        # ReLU to keep only features that have a positive influence
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        
        # Normalize to [0, 1]
        if np.max(cam) != np.min(cam):
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        else:
            cam = np.zeros_like(cam)
            
        # Resize to input resolution
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = cam_img.resize((input_tensor.shape[3], input_tensor.shape[2]), Image.BILINEAR)
        return np.array(cam_resized) / 255.0


def overlay_heatmap(orig_img, heatmap, alpha=0.45, colormap="jet"):
    """
    Superimposes the Grad-CAM heatmap over the original PIL image.
    orig_img: PIL Image
    heatmap: 2D numpy array [0, 1]
    returns: PIL Image
    """
    import matplotlib.cm as cm
    
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(orig_img.size, Image.BILINEAR)
    heatmap_arr = np.array(heatmap_resized) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_arr)[:, :, :3]  # drop alpha
    colored_heatmap = np.uint8(colored_heatmap * 255)
    
    orig_arr = np.array(orig_img.convert("RGB"))
    blended = np.uint8(orig_arr * (1 - alpha) + colored_heatmap * alpha)
    return Image.fromarray(blended)


# -----------------------------------------------------------------------------
# 5. Intermediate Feature Map Activations
# -----------------------------------------------------------------------------
def get_feature_maps(model, input_tensor):
    """
    Passes an input tensor through the model and captures activation maps
    from conv1, conv2, and conv3 (if present).
    returns: dict of {layer_name: numpy array (channels, H, W)}
    """
    model.eval()
    feature_maps = {}
    with torch.no_grad():
        x = input_tensor
        x = model.conv1(x)
        x = model.bn1(x)
        x = F.relu(x)
        feature_maps["Conv Layer 1 (Low-level Edges/Textures)"] = x.squeeze(0).cpu().numpy()
        x = model.pool1(x)
        
        x = model.conv2(x)
        x = model.bn2(x)
        x = F.relu(x)
        feature_maps["Conv Layer 2 (Mid-level Shapes/Patterns)"] = x.squeeze(0).cpu().numpy()
        x = model.pool2(x)
        
        if model.conv3 is not None:
            x = model.conv3(x)
            x = model.bn3(x)
            x = F.relu(x)
            feature_maps["Conv Layer 3 (High-level Semantic Parts)"] = x.squeeze(0).cpu().numpy()
            
    return feature_maps