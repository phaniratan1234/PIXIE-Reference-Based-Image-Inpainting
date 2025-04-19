#!/usr/bin/env python3
"""
Uncertainty estimation network for image inpainting.
Quantifies how difficult each pixel is to reconstruct based on available context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from .base import BaseModel, AttentionLayer, GatedConv2d

class UncertaintyNetwork(BaseModel):
    """Network to estimate uncertainty in inpainting predictions."""
    
    def __init__(self, config=None):
        """
        Initialize uncertainty estimation network.
        
        Args:
            config (dict, optional): Configuration for the network
                - channels (int): Number of channels in convolutional layers
                - num_blocks (int): Number of residual blocks
                - attention_heads (int): Number of attention heads
        """
        super().__init__(config)
        
        # Default configuration
        self.channels = self.config.get('channels', 64)
        self.num_blocks = self.config.get('num_blocks', 6)
        self.attention_heads = self.config.get('attention_heads', 8)
        
        # Input processing
        # Takes RGB image, mask, edge map, and semantic features (optional)
        self.input_channels = 3 + 1 + 1  # image + mask + edge map
        if 'semantic_features' in self.config:
            self.input_channels += self.config['semantic_features']
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build uncertainty estimation network.
        
        Returns:
            nn.Sequential: Uncertainty estimation model
        """
        layers = []
        
        # Initial input processing
        layers.append(
            nn.Conv2d(self.input_channels, self.channels, kernel_size=3, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Downsampling
        layers.append(
            nn.Conv2d(self.channels, self.channels * 2, kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        layers.append(
            nn.Conv2d(self.channels * 2, self.channels * 4, kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Feature processing with residual blocks and attention
        current_channels = self.channels * 4
        
        # Residual blocks
        for _ in range(self.num_blocks):
            layers.append(UncertaintyResidualBlock(current_channels))
        
        # Attention mechanism for global reasoning
        layers.append(AttentionLayer(current_channels, num_heads=self.attention_heads))
        
        # Upsampling
        layers.append(
            nn.ConvTranspose2d(current_channels, self.channels * 2, 
                              kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        layers.append(
            nn.ConvTranspose2d(self.channels * 2, self.channels, 
                              kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer for uncertainty map
        layers.append(
            nn.Conv2d(self.channels, 1, kernel_size=3, padding=1)
        )
        layers.append(nn.Sigmoid())  # Normalize to [0, 1]
        
        return nn.Sequential(*layers)
    
    def forward(self, image, mask, edge_map=None, semantic_features=None):
        """
        Forward pass of uncertainty estimation network.
        
        Args:
            image (torch.Tensor): Input image tensor [B, 3, H, W]
            mask (torch.Tensor): Binary mask tensor [B, 1, H, W]
            edge_map (torch.Tensor, optional): Edge map tensor [B, 1, H, W]
            semantic_features (torch.Tensor, optional): Semantic features
            
        Returns:
            dict: Uncertainty estimation results with keys:
                - uncertainty: Uncertainty map tensor
                - confidence: Confidence map tensor (1 - uncertainty)
        """
        batch_size, _, height, width = image.shape
        device = image.device
        
        # If edge map is not provided, create a dummy one
        if edge_map is None:
            edge_map = torch.zeros((batch_size, 1, height, width), device=device)
        
        # Concatenate inputs
        inputs = [image, mask, edge_map]
        
        # Add semantic features if provided
        if semantic_features is not None:
            # Ensure semantic features have the right dimensions
            if semantic_features.shape[2:] != (height, width):
                semantic_features = F.interpolate(
                    semantic_features, size=(height, width), mode='bilinear', align_corners=False
                )
            inputs.append(semantic_features)
        
        x = torch.cat(inputs, dim=1)
        
        # Forward pass through model
        uncertainty = self.model(x)
        
        # Apply special handling for masked regions
        # Inside the mask, uncertainty should be high
        # Outside the mask, it should be computed normally
        mask_weight = 0.5
        final_uncertainty = uncertainty * (1 - mask) + mask * mask_weight
        
        # Confidence is the opposite of uncertainty
        confidence = 1.0 - final_uncertainty
        
        return {
            'uncertainty': final_uncertainty,
            'confidence': confidence
        }
    
    def get_confidence_weighted_features(self, features, mask):
        """
        Weight features by confidence (inverse uncertainty).
        
        Args:
            features (torch.Tensor): Input feature tensor [B, C, H, W]
            mask (torch.Tensor): Binary mask tensor [B, 1, H, W]
            
        Returns:
            torch.Tensor: Confidence-weighted features
        """
        # Get raw uncertainty estimation
        uncertainty_result = self.forward(features, mask)
        confidence = uncertainty_result['confidence']
        
        # Apply confidence weighting
        weighted_features = features * confidence
        
        return weighted_features
    
    def get_uncertainty_map(self, image, mask, edge_map=None, semantic_features=None, apply_filter=True):
        """
        Get uncertainty map for visualization.
        
        Args:
            image (torch.Tensor): Input image tensor
            mask (torch.Tensor): Binary mask tensor
            edge_map (torch.Tensor, optional): Edge map tensor
            semantic_features (torch.Tensor, optional): Semantic features
            apply_filter (bool): Whether to apply bilateral filtering for smoothness
            
        Returns:
            torch.Tensor: Uncertainty map in range [0, 1]
        """
        # Get uncertainty map
        uncertainty_result = self.forward(image, mask, edge_map, semantic_features)
        uncertainty = uncertainty_result['uncertainty']
        
        if apply_filter:
            # Apply bilateral filter for smoother uncertainty maps
            # This is done in numpy as PyTorch doesn't have a direct bilateral filter
            smooth_uncertainty = torch.zeros_like(uncertainty)
            
            for i in range(uncertainty.size(0)):
                # Convert to numpy for OpenCV
                uncert_np = uncertainty[i, 0].detach().cpu().numpy()
                
                # Reference image for bilateral filter (use grayscale image)
                if image.size(1) == 3:
                    # Convert RGB to grayscale
                    img_np = image[i].permute(1, 2, 0).detach().cpu().numpy()
                    img_gray = np.mean(img_np, axis=2).astype(np.float32)
                else:
                    img_gray = image[i, 0].detach().cpu().numpy().astype(np.float32)
                
                # Apply bilateral filter
                filtered = cv2.bilateralFilter(uncert_np.astype(np.float32), 
                                              d=9, sigmaColor=75, sigmaSpace=75)
                
                # Convert back to tensor
                smooth_uncertainty[i, 0] = torch.from_numpy(filtered).to(uncertainty.device)
            
            return smooth_uncertainty
        
        return uncertainty

class UncertaintyResidualBlock(nn.Module):
    """Residual block for uncertainty estimation network."""
    
    def __init__(self, channels):
        """
        Initialize uncertainty residual block.
        
        Args:
            channels (int): Number of input/output channels
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(channels)
    
    def forward(self, x):
        """
        Forward pass of uncertainty residual block.
        
        Args:
            x (torch.Tensor): Input feature tensor
            
        Returns:
            torch.Tensor: Output feature tensor
        """
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.activation(out)
        
        return out

class UncertaintyGuidedAttention(nn.Module):
    """Attention module guided by uncertainty estimation."""
    
    def __init__(self, channels, num_heads=8):
        """
        Initialize uncertainty-guided attention.
        
        Args:
            channels (int): Number of input channels
            num_heads (int): Number of attention heads
        """
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # Attention layers
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Uncertainty weighting layer
        self.uncertainty_gate = nn.Conv2d(1, channels, kernel_size=1)
        
        # Output projection
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, uncertainty):
        """
        Forward pass of uncertainty-guided attention.
        
        Args:
            x (torch.Tensor): Input feature tensor [B, C, H, W]
            uncertainty (torch.Tensor): Uncertainty map [B, 1, H, W]
            
        Returns:
            torch.Tensor: Attention-enhanced features
        """
        batch_size, _, height, width = x.shape
        
        # Apply uncertainty weighting
        uncertainty_weights = self.uncertainty_gate(uncertainty)
        gated_x = x * (1.0 - uncertainty_weights)
        
        # Compute query, key, value
        q = self.query(gated_x)
        k = self.key(gated_x)
        v = self.value(x)  # Value uses original features
        
        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, self.head_dim, height * width)
        k = k.view(batch_size, self.num_heads, self.head_dim, height * width)
        v = v.view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # Transpose for batch matrix multiplication
        q = q.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        k = k.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        v = v.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        
        # Apply mask based on uncertainty
        uncertainty_flat = uncertainty.view(batch_size, 1, 1, height * width)
        uncertainty_mask = (1.0 - uncertainty_flat.expand(-1, self.num_heads, height * width, -1))
        
        # Scale attention scores by inverse uncertainty
        attn_scores = attn_scores * uncertainty_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        out = out.transpose(2, 3).contiguous()
        out = out.view(batch_size, self.channels, height, width)
        
        # Final projection
        out = self.output_proj(out)
        
        return out + x  # Residual connection 