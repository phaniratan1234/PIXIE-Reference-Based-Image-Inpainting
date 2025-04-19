#!/usr/bin/env python3
"""
Base classes for all neural network models in the image inpainting project.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config=None):
        """
        Initialize the base model.
        
        Args:
            config (dict, optional): Model configuration
        """
        super().__init__()
        self.config = {} if config is None else config
        self.device = torch.device('cpu')
    
    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: Target device
            
        Returns:
            BaseModel: Self
        """
        self.device = device
        return super().to(device)
    
    @abstractmethod
    def forward(self, x, **kwargs):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Model output
        """
        pass
    
    def save(self, path):
        """
        Save model checkpoint.
        
        Args:
            path (str): Path to save the checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path, strict=True):
        """
        Load model checkpoint.
        
        Args:
            path (str): Path to the checkpoint
            strict (bool): Whether to strictly enforce that the keys
                in state_dict match the keys in this model's state_dict
                
        Returns:
            BaseModel: Self
        """
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        else:
            self.load_state_dict(checkpoint, strict=strict)
        
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        print(f"Model loaded from {path}")
        return self
    
    def count_parameters(self):
        """
        Count the number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def partial_freeze(self, layers_to_freeze):
        """
        Freeze specific layers of the model.
        
        Args:
            layers_to_freeze (list): List of layer names to freeze
        """
        for name, param in self.named_parameters():
            for layer_name in layers_to_freeze:
                if layer_name in name:
                    param.requires_grad = False
                    break

class InpaintingModel(BaseModel):
    """Base class for image inpainting models."""
    
    def __init__(self, config=None):
        """
        Initialize the inpainting model.
        
        Args:
            config (dict, optional): Model configuration
        """
        super().__init__(config)
    
    @abstractmethod
    def forward(self, masked_image, mask, **kwargs):
        """
        Forward pass of the inpainting model.
        
        Args:
            masked_image (torch.Tensor): Image with masked region
            mask (torch.Tensor): Binary mask indicating masked regions
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Completed image
        """
        pass
    
    def predict(self, masked_image, mask, **kwargs):
        """
        Generate a prediction for the masked regions.
        
        Args:
            masked_image (torch.Tensor): Image with masked region
            mask (torch.Tensor): Binary mask indicating masked regions
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Completed image
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(masked_image, mask, **kwargs)
        return result
    
    def get_training_losses(self, batch, **kwargs):
        """
        Calculate training losses.
        
        Args:
            batch (dict): Batch of data
            **kwargs: Additional arguments
            
        Returns:
            dict: Dictionary of loss values
        """
        raise NotImplementedError("Subclasses must implement this method")

class AttentionLayer(nn.Module):
    """Multi-head attention layer for feature enhancement."""
    
    def __init__(self, channels, num_heads=8, dropout=0.0):
        """
        Initialize attention layer.
        
        Args:
            channels (int): Number of input channels
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Projection layers
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        """
        Forward pass of attention layer.
        
        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Enhanced feature map
        """
        batch_size, _, height, width = x.shape
        
        # Compute query, key, value projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, self.head_dim, height * width)
        k = k.view(batch_size, self.num_heads, self.head_dim, height * width)
        v = v.view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # Transpose for batch matrix multiplication
        q = q.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        k = k.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        v = v.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale  # [B, num_heads, H*W, H*W]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, height * width)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, num_heads, H*W, head_dim]
        
        # Transpose and reshape back
        out = out.transpose(2, 3).contiguous()  # [B, num_heads, head_dim, H*W]
        out = out.view(batch_size, self.channels, height, width)
        
        # Final projection
        out = self.output(out)
        
        return out

class GatedConv2d(nn.Module):
    """Gated convolution layer for masked inputs."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, activation=nn.LeakyReLU(0.2)):
        """
        Initialize gated convolution layer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size
            stride (int): Convolution stride
            padding (int): Padding size
            dilation (int): Dilation rate
            groups (int): Number of groups
            bias (bool): Whether to use bias
            activation: Activation function
        """
        super().__init__()
        
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.sigmoid = nn.Sigmoid()
        self.activation = activation
    
    def forward(self, x, mask=None):
        """
        Forward pass of gated convolution.
        
        Args:
            x (torch.Tensor): Input feature map
            mask (torch.Tensor, optional): Input mask (1 for valid, 0 for masked)
            
        Returns:
            torch.Tensor: Output feature map
        """
        features = self.conv2d(x)
        
        if mask is not None:
            mask_features = self.mask_conv2d(x * mask) * (1 - mask) + self.mask_conv2d(x) * mask
        else:
            mask_features = self.mask_conv2d(x)
        
        gate = self.sigmoid(mask_features)
        output = features * gate
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output

class ResnetBlock(nn.Module):
    """Basic ResNet block with optional instance normalization."""
    
    def __init__(self, channels, kernel_size=3, dilation=1, use_spectral_norm=False, use_norm=True):
        """
        Initialize ResNet block.
        
        Args:
            channels (int): Number of channels
            kernel_size (int): Kernel size
            dilation (int): Dilation rate
            use_spectral_norm (bool): Whether to use spectral normalization
            use_norm (bool): Whether to use instance normalization
        """
        super().__init__()
        
        padding = kernel_size // 2 * dilation
        
        # Conv layers
        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(
                nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation))
            self.conv2 = nn.utils.spectral_norm(
                nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation))
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        
        # Normalization layers
        if use_norm:
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        """
        Forward pass of ResNet block.
        
        Args:
            x (torch.Tensor): Input feature map
            
        Returns:
            torch.Tensor: Output feature map
        """
        residual = x
        
        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)
        
        return out + residual 