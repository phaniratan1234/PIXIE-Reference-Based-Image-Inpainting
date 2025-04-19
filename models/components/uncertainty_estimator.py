"""
Uncertainty Estimation Module

This module provides functionality for estimating uncertainty in inpainting predictions,
allowing the model to quantify regions where predictions might be less reliable.
The uncertainty maps can be used to guide refinement stages and weight loss functions.

The module supports multiple types of uncertainty estimation:
- Mask uncertainty: Confidence in the mask boundary regions
- Structure uncertainty: Confidence in structural predictions
- Texture uncertainty: Confidence in texture details
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Dict, Optional, List, Tuple, Any, Union
from einops import rearrange
import numpy as np

class UncertaintyEstimator(nn.Module):
    """
    A module for estimating uncertainty in inpainting predictions.
    
    This estimator generates uncertainty maps for different aspects of the inpainting
    process, which can be used to:
    1. Guide the refinement of predicted content
    2. Weight loss functions during training (lower weight for high uncertainty regions)
    3. Provide user feedback on prediction reliability
    
    The estimator uses a multi-scale architecture with attention mechanisms to
    identify regions of varying uncertainty levels.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        uncertainty_types: Union[List[str], str, bytes, int] = ['mask', 'structure', 'texture'],
        num_scales: int = 3,
        use_attention: bool = True,
        attention_heads: int = 4,
        use_spectral_norm: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ):
        """
        Initialize the uncertainty estimator.
        
        Args:
            in_channels: Number of input feature channels
            hidden_channels: Number of hidden feature channels in the network
            uncertainty_types: Types of uncertainty to estimate. Can be:
                - List of strings: e.g., ['mask', 'structure', 'texture']
                - Single string: comma-separated values, e.g., "mask,structure,texture"
                - Bytes: UTF-8 encoded string
                - Integer: Number of uncertainty types to use (uses default types)
                - None: Uses all default types
                Default types include 'mask', 'structure', and 'texture'
            num_scales: Number of scales for multi-scale processing
            use_attention: Whether to use attention mechanisms
            attention_heads: Number of attention heads if attention is used
            use_spectral_norm: Whether to use spectral normalization for stability
            device: Device to initialize the model on (e.g., 'cuda', 'cpu')
            **kwargs: Additional arguments to pass to submodules
        """
        super().__init__()
        
        # Process and store configuration
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_scales = num_scales
        self.use_attention = use_attention
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Process uncertainty types from various input formats
        if uncertainty_types is None:
            self.uncertainty_types = ['mask', 'structure', 'texture']
        elif isinstance(uncertainty_types, int):
            # If integer is provided, use that many default uncertainty types
            default_types = ['mask', 'structure', 'texture']
            self.uncertainty_types = default_types[:min(uncertainty_types, len(default_types))]
        elif isinstance(uncertainty_types, bytes):
            # Handle bytes input (convert to string and split)
            self.uncertainty_types = uncertainty_types.decode('utf-8').lower().split(',')
        elif isinstance(uncertainty_types, str):
            # Handle string input (split by comma)
            self.uncertainty_types = uncertainty_types.lower().split(',')
        else:
            # Handle list input
            self.uncertainty_types = [utype.lower() for utype in uncertainty_types]
        
        # Feature extraction network (shared across uncertainty types)
        norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Encoder blocks for feature extraction
        self.encoder = nn.ModuleList([
            nn.Sequential(
                norm_layer(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2),
                norm_layer(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2)
            )
        ])
        
        # Add downsampling blocks if using multiple scales
        for i in range(1, num_scales):
            in_ch = hidden_channels
            out_ch = hidden_channels * 2 if i < num_scales - 1 else hidden_channels
            self.encoder.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=2),  # Downsample
                norm_layer(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2),
                norm_layer(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2)
            ))
        
        # Attention module for capturing global context
        if use_attention:
            self.attention = self.AttentionModule(
                hidden_channels, 
                heads=attention_heads,
                use_spectral_norm=use_spectral_norm
            )
        
        # Uncertainty heads - one for each uncertainty type
        # Use ModuleDict for named access to different heads
        self.uncertainty_heads = nn.ModuleDict()
        for utype in self.uncertainty_types:
            self.uncertainty_heads[utype] = nn.Sequential(
                norm_layer(nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2),
                norm_layer(nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, padding=1)),
                nn.Sigmoid()  # Normalize uncertainty to [0, 1] range
            )
        
        # Move model to the specified device
        self.to(self.device)
    
    def forward(
        self, 
        features: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        structure: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate uncertainty and confidence maps for the input features.
        
        Args:
            features: Input feature maps [B, C, H, W]
            masks: Binary masks indicating regions to inpaint [B, 1, H, W]
                  1 = area to inpaint (hole), 0 = known region
            structure: Structural information [B, C, H, W], if available
            **kwargs: Additional inputs that may be used by some uncertainty types
        
        Returns:
            Dictionary containing:
                'uncertainty_maps': Dict mapping uncertainty type to corresponding map
                'confidence_maps': Dict mapping uncertainty type to confidence map (1 - uncertainty)
        """
        # Move inputs to the correct device
        features = features.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
        
        batch_size, _, height, width = features.shape
        
        # Process features through encoder blocks
        feature_maps = []
        x = features
        for block in self.encoder:
            x = block(x)
            feature_maps.append(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Include mask information if available
        if masks is not None and 'mask' in self.uncertainty_types:
            # Resize masks to match feature resolution if needed
            if masks.shape[2:] != x.shape[2:]:
                resized_masks = F.interpolate(
                    masks, 
                    size=x.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                # Concatenate mask information
                x = torch.cat([x, resized_masks], dim=1)
                # Adjust channels back to hidden_channels using a 1x1 conv
                x = nn.Conv2d(
                    x.shape[1], 
                    self.hidden_channels, 
                    kernel_size=1
                ).to(self.device)(x)
        
        # Include structure information if available and needed
        if structure is not None and 'structure' in self.uncertainty_types:
            # Resize structure to match feature resolution if needed
            if structure.shape[2:] != x.shape[2:]:
                resized_structure = F.interpolate(
                    structure,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                # Process structure information with a small network
                structure_features = nn.Sequential(
                    nn.Conv2d(structure.shape[1], self.hidden_channels // 2, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_channels // 2, self.hidden_channels // 2, kernel_size=3, padding=1),
                ).to(self.device)(resized_structure)
                
                # Concatenate structure features
                x = torch.cat([x, structure_features], dim=1)
                # Adjust channels back to hidden_channels
                x = nn.Conv2d(
                    x.shape[1], 
                    self.hidden_channels, 
                    kernel_size=1
                ).to(self.device)(x)
        
        # Generate uncertainty maps for each uncertainty type
        uncertainty_maps = {}
        confidence_maps = {}
        
        for utype in self.uncertainty_types:
            # Apply the corresponding uncertainty head
            if utype in self.uncertainty_heads:
                uncertainty = self.uncertainty_heads[utype](x)
                
                # Resize to original input resolution
                if uncertainty.shape[2:] != (height, width):
                    uncertainty = F.interpolate(
                        uncertainty,
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Store uncertainty map
                uncertainty_maps[utype] = uncertainty
                
                # Compute confidence map (1 - uncertainty)
                confidence_maps[utype] = 1.0 - uncertainty
        
        # Return both uncertainty and confidence maps
        return {
            'uncertainty_maps': uncertainty_maps,
            'confidence_maps': confidence_maps
        }
    
    class AttentionModule(nn.Module):
        """
        Multi-head self-attention module for capturing global dependencies.
        
        This module allows the uncertainty estimator to consider the entire image
        context when determining uncertainty levels, improving the accuracy of
        uncertainty maps, especially at object boundaries and complex regions.
        """
        
        def __init__(self, channels: int, heads: int = 4, use_spectral_norm: bool = False):
            """
            Initialize the attention module.
            
            Args:
                channels: Number of input/output channels
                heads: Number of attention heads
                use_spectral_norm: Whether to use spectral normalization
            """
            super().__init__()
            
            self.heads = heads
            self.channels = channels
            self.scale = (channels // heads) ** -0.5  # Scaling factor for dot product
            
            # Normalization function based on configuration
            norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
            
            # Query, key, value projections
            self.to_q = norm_layer(nn.Conv1d(channels, channels, 1))
            self.to_k = norm_layer(nn.Conv1d(channels, channels, 1))
            self.to_v = norm_layer(nn.Conv1d(channels, channels, 1))
            
            # Output projection
            self.to_out = nn.Sequential(
                norm_layer(nn.Conv1d(channels, channels, 1)),
                nn.LeakyReLU(0.2)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Apply self-attention to the input features.
            
            Args:
                x: Input features [B, C, H, W]
            
            Returns:
                Attention-refined features [B, C, H, W]
            """
            batch_size, channels, height, width = x.shape
            
            # Reshape to [B, C, H*W]
            x_flat = x.view(batch_size, channels, -1)
            
            # Project to queries, keys, values
            q = self.to_q(x_flat)
            k = self.to_k(x_flat)
            v = self.to_v(x_flat)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, self.heads, channels // self.heads, -1)
            k = k.view(batch_size, self.heads, channels // self.heads, -1)
            v = v.view(batch_size, self.heads, channels // self.heads, -1)
            
            # Transpose for batch matrix multiplication
            q = q.transpose(-2, -1)  # [B, heads, HW, C//heads]
            k = k.transpose(-2, -1)  # [B, heads, HW, C//heads]
            v = v.transpose(-2, -1)  # [B, heads, HW, C//heads]
            
            # Compute attention scores
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, HW, HW]
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to values
            out = torch.matmul(attn, v)  # [B, heads, HW, C//heads]
            
            # Reshape back
            out = out.transpose(-2, -1).contiguous()  # [B, heads, C//heads, HW]
            out = out.view(batch_size, channels, -1)  # [B, C, HW]
            
            # Apply output projection
            out = self.to_out(out)
            
            # Reshape to original dimensions
            out = out.view(batch_size, channels, height, width)
            
            # Residual connection
            return out + x 