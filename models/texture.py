#!/usr/bin/env python3
"""
Texture Artist Network for image inpainting.
Responsible for adding realistic surface details and colors to the structural framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from .base import BaseModel, AttentionLayer, GatedConv2d, ResnetBlock

class TextureArtist(BaseModel):
    """Texture artist network for generating realistic textures and colors."""
    
    def __init__(self, config=None):
        """
        Initialize texture artist network.
        
        Args:
            config (dict, optional): Configuration for the network
                - channels (int): Base number of channels
                - num_blocks (int): Number of residual blocks
                - style_transfer (bool): Whether to use style transfer mechanisms
                - noise_injection (bool): Whether to inject noise for texture variety
        """
        super().__init__(config)
        
        # Default configuration
        self.channels = self.config.get('channels', 128)
        self.num_blocks = self.config.get('num_blocks', 8)
        self.style_transfer = self.config.get('style_transfer', True)
        self.noise_injection = self.config.get('noise_injection', True)
        
        # Build model
        self.encoder = self._build_encoder()
        self.bottleneck = self._build_bottleneck()
        self.decoder = self._build_decoder()
        
        # Style transfer module (if enabled)
        if self.style_transfer:
            self.style_encoder = self._build_style_encoder()
            self.cross_attention = self._build_cross_attention()
        
        # Output layers
        self.to_rgb = nn.Sequential(
            nn.Conv2d(self.channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def _build_encoder(self):
        """
        Build encoder part of the network.
        
        Returns:
            nn.ModuleList: List of encoder blocks
        """
        encoder_blocks = nn.ModuleList()
        
        # Initial input: Image, mask, structure map, edge map (6 channels)
        input_channels = 6
        
        # Initial convolution (no downsampling)
        encoder_blocks.append(nn.Sequential(
            GatedConv2d(input_channels, self.channels, kernel_size=7, padding=3),
            nn.InstanceNorm2d(self.channels),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        # Downsampling blocks
        num_downsample = 3
        current_channels = self.channels
        
        for i in range(num_downsample):
            next_channels = current_channels * 2
            encoder_blocks.append(nn.Sequential(
                GatedConv2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            current_channels = next_channels
        
        return encoder_blocks
    
    def _build_bottleneck(self):
        """
        Build bottleneck part of the network.
        
        Returns:
            nn.ModuleList: List of bottleneck blocks
        """
        bottleneck_blocks = nn.ModuleList()
        
        # Current channels after encoder
        current_channels = self.channels * (2 ** 3)
        
        # Residual blocks
        for i in range(self.num_blocks):
            bottleneck_blocks.append(ResnetBlock(current_channels))
            
            # Add attention layer every 2 blocks
            if i % 2 == 1:
                bottleneck_blocks.append(AttentionLayer(current_channels))
        
        return bottleneck_blocks
    
    def _build_decoder(self):
        """
        Build decoder part of the network.
        
        Returns:
            nn.ModuleList: List of decoder blocks
        """
        decoder_blocks = nn.ModuleList()
        
        # Current channels after bottleneck
        current_channels = self.channels * (2 ** 3)
        
        # Upsampling blocks with skip connections
        num_upsample = 3
        
        for i in range(num_upsample):
            next_channels = current_channels // 2
            
            decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(current_channels, next_channels, 
                                  kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            
            # Add AdaIN residual block if style transfer is enabled
            if self.style_transfer:
                decoder_blocks.append(AdaINResBlock(next_channels))
            else:
                decoder_blocks.append(ResnetBlock(next_channels))
            
            # Inject noise for texture variety if enabled
            if self.noise_injection and i < num_upsample - 1:
                decoder_blocks.append(NoiseInjection(next_channels))
            
            current_channels = next_channels
            
            # Add attention on last decoder layer
            if i == num_upsample - 1:
                decoder_blocks.append(AttentionLayer(current_channels))
        
        return decoder_blocks
    
    def _build_style_encoder(self):
        """
        Build style encoder for exemplar images.
        
        Returns:
            nn.Sequential: Style encoder model
        """
        layers = []
        
        # Initial convolution
        layers.append(nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3))
        layers.append(nn.InstanceNorm2d(64))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Downsampling
        in_channels = 64
        for i in range(4):
            out_channels = min(in_channels * 2, 512)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_cross_attention(self):
        """
        Build cross-attention module for style transfer.
        
        Returns:
            nn.Module: Cross-attention module
        """
        # Calculate feature dimensions after style encoder
        style_channels = min(64 * (2 ** 4), 512)
        content_channels = self.channels * (2 ** 3)
        
        return CrossAttention(content_channels, style_channels)
    
    def forward(self, masked_image, mask, structure_map, edge_map, exemplar=None):
        """
        Forward pass of texture artist network.
        
        Args:
            masked_image (torch.Tensor): Input masked image tensor [B, 3, H, W]
            mask (torch.Tensor): Binary mask tensor [B, 1, H, W]
            structure_map (torch.Tensor): Structure map tensor [B, 1, H, W]
            edge_map (torch.Tensor): Edge map tensor [B, 1, H, W]
            exemplar (torch.Tensor, optional): Exemplar image tensor [B, 3, H, W]
            
        Returns:
            dict: Texture artist results with keys:
                - texture: Generated RGB texture/color output
                - composite: Final composite image with background from masked_image
        """
        batch_size, _, height, width = masked_image.shape
        device = masked_image.device
        
        # Concatenate inputs
        x = torch.cat([masked_image, mask, structure_map, edge_map], dim=1)
        
        # Encoder with skip connections
        encoder_features = []
        for block in self.encoder:
            x = block(x)
            encoder_features.append(x)
        
        # Process style features if style transfer is enabled and exemplar is provided
        style_features = None
        if self.style_transfer and exemplar is not None:
            style_features = self.style_encoder(exemplar)
            
            # Apply cross-attention for style transfer
            x = self.cross_attention(x, style_features)
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x)
        
        # Decoder with skip connections
        for i, block in enumerate(self.decoder):
            # Apply skip connection at the start of each upsampling block
            if isinstance(block[0], nn.ConvTranspose2d) and encoder_features:
                skip_features = encoder_features.pop()
                x = torch.cat([x, skip_features], dim=1)
                
                # Adjust channels if needed
                if x.shape[1] != block[0].in_channels:
                    channel_adapter = nn.Conv2d(
                        x.shape[1], block[0].in_channels, kernel_size=1
                    ).to(device)
                    x = channel_adapter(x)
            
            # Apply style transfer for AdaIN blocks
            if isinstance(block, AdaINResBlock) and style_features is not None:
                # Extract style statistics for AdaIN
                style_mean, style_std = calc_mean_std(style_features)
                x = block(x, style_mean, style_std)
            else:
                x = block(x)
        
        # Generate RGB output
        rgb_output = self.to_rgb(x)
        
        # Create composite output (copy background from masked_image)
        valid_region = 1.0 - mask
        composite = masked_image * valid_region + rgb_output * mask
        
        return {
            'texture': rgb_output,
            'composite': composite
        }

class CrossAttention(nn.Module):
    """Cross-attention module for style transfer between content and style features."""
    
    def __init__(self, content_channels, style_channels, num_heads=8):
        """
        Initialize cross-attention module.
        
        Args:
            content_channels (int): Number of content feature channels
            style_channels (int): Number of style feature channels
            num_heads (int): Number of attention heads
        """
        super().__init__()
        
        self.num_heads = num_heads
        
        # Project style features to match content feature dimensions
        self.style_proj = nn.Conv2d(style_channels, content_channels, kernel_size=1)
        
        # Query, key, value projections
        self.query_proj = nn.Conv2d(content_channels, content_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(content_channels, content_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(content_channels, content_channels, kernel_size=1)
        
        # Output projection
        self.output_proj = nn.Conv2d(content_channels, content_channels, kernel_size=1)
        
        # Scaling factor
        self.scale = (content_channels // num_heads) ** -0.5
    
    def forward(self, content_features, style_features):
        """
        Forward pass of cross-attention module.
        
        Args:
            content_features (torch.Tensor): Content features [B, C, H, W]
            style_features (torch.Tensor): Style features [B, S, Hs, Ws]
            
        Returns:
            torch.Tensor: Style-enhanced content features
        """
        batch_size, channels, height, width = content_features.shape
        
        # Project style features to match content dimensions
        style_features = self.style_proj(style_features)
        
        # Resize style features to match content feature size if needed
        if style_features.shape[2:] != content_features.shape[2:]:
            style_features = F.interpolate(
                style_features, size=(height, width), 
                mode='bilinear', align_corners=False
            )
        
        # Calculate query from content, key/value from style
        q = self.query_proj(content_features)
        k = self.key_proj(style_features)
        v = self.value_proj(style_features)
        
        # Reshape for multi-head attention
        head_dim = channels // self.num_heads
        q = q.view(batch_size, self.num_heads, head_dim, height * width)
        k = k.view(batch_size, self.num_heads, head_dim, height * width)
        v = v.view(batch_size, self.num_heads, head_dim, height * width)
        
        # Transpose for batch matrix multiplication
        q = q.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        k = k.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        v = v.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        out = out.transpose(2, 3).contiguous()
        out = out.view(batch_size, channels, height, width)
        
        # Final projection
        out = self.output_proj(out)
        
        # Residual connection
        out = out + content_features
        
        return out

class AdaINResBlock(nn.Module):
    """Residual block with Adaptive Instance Normalization for style transfer."""
    
    def __init__(self, channels):
        """
        Initialize AdaIN residual block.
        
        Args:
            channels (int): Number of input/output channels
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x, style_mean=None, style_std=None):
        """
        Forward pass of AdaIN residual block.
        
        Args:
            x (torch.Tensor): Input feature tensor
            style_mean (torch.Tensor, optional): Style mean for AdaIN
            style_std (torch.Tensor, optional): Style std for AdaIN
            
        Returns:
            torch.Tensor: Output feature tensor
        """
        residual = x
        
        # First convolution
        out = self.conv1(x)
        
        # Apply AdaIN if style statistics are provided
        if style_mean is not None and style_std is not None:
            out = adaptive_instance_norm(out, style_mean, style_std)
        else:
            # Fallback to regular instance norm
            out = F.instance_norm(out)
        
        out = self.activation(out)
        
        # Second convolution
        out = self.conv2(out)
        
        # Apply AdaIN if style statistics are provided
        if style_mean is not None and style_std is not None:
            out = adaptive_instance_norm(out, style_mean, style_std)
        else:
            # Fallback to regular instance norm
            out = F.instance_norm(out)
        
        # Residual connection
        out = out + residual
        out = self.activation(out)
        
        return out

class NoiseInjection(nn.Module):
    """Noise injection module for texture variety."""
    
    def __init__(self, channels):
        """
        Initialize noise injection module.
        
        Args:
            channels (int): Number of feature channels
        """
        super().__init__()
        
        # Learnable noise scale factors for each channel
        self.noise_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x):
        """
        Forward pass of noise injection module.
        
        Args:
            x (torch.Tensor): Input feature tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Noise-enhanced feature tensor
        """
        if self.training:
            batch_size, _, height, width = x.shape
            
            # Generate random noise
            noise = torch.randn(batch_size, 1, height, width, device=x.device)
            
            # Scale noise per-channel and add to input
            return x + self.noise_scale * noise
        else:
            return x

def calc_mean_std(features):
    """
    Calculate mean and standard deviation for AdaIN.
    
    Args:
        features (torch.Tensor): Feature tensor [B, C, H, W]
        
    Returns:
        tuple: (mean, std) tensors with shape [B, C, 1, 1]
    """
    batch_size, channels = features.shape[:2]
    
    # Flatten spatial dimensions
    features_reshaped = features.view(batch_size, channels, -1)
    
    # Calculate mean and std along spatial dimensions
    mean = features_reshaped.mean(dim=2).view(batch_size, channels, 1, 1)
    std = features_reshaped.std(dim=2).view(batch_size, channels, 1, 1) + 1e-5
    
    return mean, std

def adaptive_instance_norm(content, style_mean, style_std):
    """
    Apply Adaptive Instance Normalization (AdaIN).
    
    Args:
        content (torch.Tensor): Content feature tensor
        style_mean (torch.Tensor): Style mean tensor
        style_std (torch.Tensor): Style std tensor
        
    Returns:
        torch.Tensor: Style-normalized content features
    """
    # Calculate content mean and std
    content_mean, content_std = calc_mean_std(content)
    
    # Normalize content features and rescale with style statistics
    normalized = (content - content_mean) / content_std
    return normalized * style_std + style_mean 