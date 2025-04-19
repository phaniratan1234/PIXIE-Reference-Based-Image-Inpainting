#!/usr/bin/env python3
"""
Structure Builder Network for image inpainting.
Responsible for generating the structural framework of inpainted regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from .base import BaseModel, AttentionLayer, GatedConv2d, ResnetBlock

class StructureBuilder(BaseModel):
    """Structure builder network for generating the structural framework of inpainted regions."""
    
    def __init__(self, config=None):
        """
        Initialize structure builder network.
        
        Args:
            config (dict, optional): Configuration for the network
                - channels (int): Base number of channels
                - num_blocks (int): Number of residual blocks
                - attention_enabled (bool): Whether to use attention layers
                - multi_scale (bool): Whether to use multi-scale feature fusion
        """
        super().__init__(config)
        
        # Default configuration
        self.channels = self.config.get('channels', 128)
        self.num_blocks = self.config.get('num_blocks', 8)
        self.attention_enabled = self.config.get('attention_enabled', True)
        self.multi_scale = self.config.get('multi_scale', True)
        
        # Build model
        self.encoder = self._build_encoder()
        self.bottleneck = self._build_bottleneck()
        self.decoder = self._build_decoder()
        
        # Output layers
        self.to_structure = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def _build_encoder(self):
        """
        Build encoder part of the network.
        
        Returns:
            nn.ModuleList: List of encoder blocks
        """
        encoder_blocks = nn.ModuleList()
        
        # Initial convolutional layer (no downsampling)
        # Input: masked image, mask, and edge map (5 channels)
        encoder_blocks.append(nn.Sequential(
            GatedConv2d(5, self.channels, kernel_size=7, padding=3),
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
        
        # Residual blocks with attention
        for i in range(self.num_blocks):
            bottleneck_blocks.append(ResnetBlock(current_channels))
            
            # Add attention layer every 2 blocks if enabled
            if self.attention_enabled and i % 2 == 1:
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
            
            # Add residual block after each upsampling
            decoder_blocks.append(ResnetBlock(next_channels))
            
            current_channels = next_channels
            
            # Add attention on last decoder layer if enabled
            if self.attention_enabled and i == num_upsample - 1:
                decoder_blocks.append(AttentionLayer(current_channels))
        
        return decoder_blocks
    
    def forward(self, masked_image, mask, edge_map=None, uncertainty_map=None):
        """
        Forward pass of structure builder network.
        
        Args:
            masked_image (torch.Tensor): Input masked image tensor [B, 3, H, W]
            mask (torch.Tensor): Binary mask tensor [B, 1, H, W]
            edge_map (torch.Tensor, optional): Edge map tensor [B, 1, H, W]
            uncertainty_map (torch.Tensor, optional): Uncertainty map tensor [B, 1, H, W]
            
        Returns:
            dict: Structure builder results with keys:
                - structure: Generated structural map
                - features: Feature maps for further processing
        """
        batch_size, _, height, width = masked_image.shape
        device = masked_image.device
        
        # Create default edge map if not provided
        if edge_map is None:
            edge_map = torch.zeros((batch_size, 1, height, width), device=device)
        
        # Concatenate inputs
        x = torch.cat([masked_image, mask, edge_map], dim=1)
        
        # Encoder with skip connections
        encoder_features = []
        for block in self.encoder:
            x = block(x)
            encoder_features.append(x)
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x)
        
        # Apply uncertainty weighting if provided
        if uncertainty_map is not None:
            # Downsample uncertainty map to match feature map size
            uncertainty_resized = F.interpolate(
                uncertainty_map, size=x.shape[2:], mode='bilinear', align_corners=False
            )
            
            # Weight features by inverse uncertainty
            confidence = 1.0 - uncertainty_resized
            x = x * confidence
        
        # Decoder with skip connections
        for i, block in enumerate(self.decoder):
            # Apply skip connection at the start of each upsampling block
            if isinstance(block[0], nn.ConvTranspose2d) and encoder_features:
                skip_features = encoder_features.pop()
                
                # If multi-scale mode is enabled, use all previous encoder features
                if self.multi_scale and len(encoder_features) > 0:
                    # Resize and combine encoder features
                    combined_features = torch.zeros_like(skip_features)
                    for feat in encoder_features:
                        # Resize to match current feature size
                        resized_feat = F.interpolate(
                            feat, size=skip_features.shape[2:], 
                            mode='bilinear', align_corners=False
                        )
                        
                        # Apply 1x1 convolution to match channels
                        if resized_feat.shape[1] != skip_features.shape[1]:
                            channel_adapter = nn.Conv2d(
                                resized_feat.shape[1], skip_features.shape[1], kernel_size=1
                            ).to(device)
                            resized_feat = channel_adapter(resized_feat)
                        
                        combined_features = combined_features + resized_feat
                    
                    # Add to skip features
                    skip_features = skip_features + 0.1 * combined_features
                
                x = torch.cat([x, skip_features], dim=1)
                
                # Adjust channels if needed
                if x.shape[1] != block[0].in_channels:
                    channel_adapter = nn.Conv2d(
                        x.shape[1], block[0].in_channels, kernel_size=1
                    ).to(device)
                    x = channel_adapter(x)
            
            x = block(x)
        
        # Generate structure output
        structure = self.to_structure(x)
        
        return {
            'structure': structure,
            'features': x
        }

class BoundaryAwareBlock(nn.Module):
    """Boundary-aware convolutional block with edge detection and gating."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        Initialize boundary-aware block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size for convolution
            stride (int): Stride for convolution
            padding (int): Padding for convolution
        """
        super().__init__()
        
        # Edge detection kernel (Sobel operator)
        self.edge_detect = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.edge_detect.weight, 0.0)
        
        # Horizontal Sobel filter
        self.edge_detect.weight.data[0, 0, 0, 0] = -1.0
        self.edge_detect.weight.data[0, 0, 0, 2] = 1.0
        self.edge_detect.weight.data[0, 0, 1, 0] = -2.0
        self.edge_detect.weight.data[0, 0, 1, 2] = 2.0
        self.edge_detect.weight.data[0, 0, 2, 0] = -1.0
        self.edge_detect.weight.data[0, 0, 2, 2] = 1.0
        
        # Vertical Sobel filter
        self.edge_detect.weight.data[1, 0, 0, 0] = -1.0
        self.edge_detect.weight.data[1, 0, 0, 1] = -2.0
        self.edge_detect.weight.data[1, 0, 0, 2] = -1.0
        self.edge_detect.weight.data[1, 0, 2, 0] = 1.0
        self.edge_detect.weight.data[1, 0, 2, 1] = 2.0
        self.edge_detect.weight.data[1, 0, 2, 2] = 1.0
        
        # Main convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Edge-aware gating
        self.edge_gate = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Activation
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        """
        Forward pass of boundary-aware block.
        
        Args:
            x (torch.Tensor): Input feature tensor
            
        Returns:
            torch.Tensor: Processed feature tensor
        """
        # Edge detection
        edges = self.edge_detect(x)
        edge_magnitude = torch.sqrt(edges[:, 0:1]**2 + edges[:, 1:2]**2)
        
        # Apply main convolution
        conv_out = self.conv(x)
        
        # Apply edge-aware gating
        edge_weights = self.edge_gate(edges)
        
        # Weight features by edge importance
        gated_out = conv_out * edge_weights
        
        # Apply activation
        out = self.activation(gated_out)
        
        return out

class AttentionAugmentedConv(nn.Module):
    """Attention-augmented convolutional layer."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 attention_heads=8, attention_ratio=0.5):
        """
        Initialize attention-augmented convolution.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size for convolution
            stride (int): Stride for convolution
            padding (int): Padding for convolution
            attention_heads (int): Number of attention heads
            attention_ratio (float): Ratio of attention channels to total output channels
        """
        super().__init__()
        
        # Split channels between convolution and attention
        self.attention_channels = int(out_channels * attention_ratio)
        self.conv_channels = out_channels - self.attention_channels
        
        # Standard convolution
        self.conv = nn.Conv2d(in_channels, self.conv_channels, 
                             kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Attention mechanism
        if self.attention_channels > 0:
            self.attention = AttentionLayer(in_channels, num_heads=attention_heads)
            self.attention_proj = nn.Conv2d(in_channels, self.attention_channels, kernel_size=1)
        
        # Activation
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        """
        Forward pass of attention-augmented convolution.
        
        Args:
            x (torch.Tensor): Input feature tensor
            
        Returns:
            torch.Tensor: Output feature tensor
        """
        # Apply standard convolution
        conv_out = self.conv(x)
        
        # Apply attention if enabled
        if self.attention_channels > 0:
            # Process with attention
            attention_out = self.attention(x)
            attention_out = self.attention_proj(attention_out)
            
            # Concatenate convolution and attention outputs
            out = torch.cat([conv_out, attention_out], dim=1)
        else:
            out = conv_out
        
        # Apply activation
        out = self.activation(out)
        
        return out 