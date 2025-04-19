"""
Structure Builder Module

This module provides functionality for generating structural information for inpainting.
The StructureBuilder is responsible for constructing coherent structural representations
of missing regions, serving as a foundation for subsequent texture generation.

The module uses a combination of:
- Multi-scale feature encoding
- Contextual attention mechanisms
- Progressive refinement of structural elements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Dict, Optional, List, Tuple, Any, Union

class StructureBuilder(nn.Module):
    """
    StructureBuilder generates coherent structural representations for inpainting.
    
    This component is responsible for creating a foundational structure in masked regions
    before detailed texture is applied. It focuses on:
    
    1. Edge continuity across mask boundaries
    2. Structural coherence with surrounding content
    3. Generation of plausible object structures within masked regions
    4. Semantic consistency with the overall scene
    
    The builder uses multiple attention mechanisms to ensure that generated structures
    align naturally with the existing content, creating a seamless foundation.
    """
    
    def __init__(
        self,
        in_channels: int,
        feature_channels: int = 64,
        attention_channels: int = 128,
        num_layers: int = 4,
        use_spectral_norm: bool = True,
        activation: str = 'relu',
        **kwargs
    ):
        """
        Initialize the structure builder.
        
        Args:
            in_channels: Number of input channels from previous stages
            feature_channels: Number of channels in the internal feature representation
            attention_channels: Number of channels in the attention mechanisms
            num_layers: Number of processing layers in each block
            use_spectral_norm: Whether to use spectral normalization for training stability
            activation: Activation function to use ('relu', 'lrelu', 'elu')
            **kwargs: Additional arguments including 'device' specification
        """
        super().__init__()
        
        # Store configuration
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.attention_channels = attention_channels
        self.num_layers = num_layers
        self.use_spectral_norm = use_spectral_norm
        
        # Set up activation function based on parameter
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Define normalization function based on configuration
        self.norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Initial feature encoding block
        self.encoder = nn.Sequential(
            self.norm_layer(nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(feature_channels),
            self.activation,
            self.norm_layer(nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(feature_channels),
            self.activation
        )
        
        # Downsampling blocks for multi-scale processing
        self.downsample = nn.Sequential(
            self.norm_layer(nn.Conv2d(feature_channels, feature_channels*2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(feature_channels*2),
            self.activation,
            self.norm_layer(nn.Conv2d(feature_channels*2, feature_channels*4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(feature_channels*4),
            self.activation
        )
        
        # Global context block with dilated convolutions for expanded receptive field
        self.global_context = nn.Sequential(
            self.norm_layer(nn.Conv2d(feature_channels*4, feature_channels*4, kernel_size=3, padding=2, dilation=2)),
            nn.InstanceNorm2d(feature_channels*4),
            self.activation,
            self.norm_layer(nn.Conv2d(feature_channels*4, feature_channels*4, kernel_size=3, padding=4, dilation=4)),
            nn.InstanceNorm2d(feature_channels*4),
            self.activation,
            self.norm_layer(nn.Conv2d(feature_channels*4, feature_channels*4, kernel_size=3, padding=8, dilation=8)),
            nn.InstanceNorm2d(feature_channels*4),
            self.activation
        )
        
        # Upsampling blocks for recovering spatial resolution
        self.upsample = nn.Sequential(
            self.norm_layer(nn.ConvTranspose2d(feature_channels*4, feature_channels*2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(feature_channels*2),
            self.activation,
            self.norm_layer(nn.ConvTranspose2d(feature_channels*2, feature_channels, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(feature_channels),
            self.activation
        )
        
        # Attention module for focusing on relevant contextual elements
        self.attention = ContextualAttention(
            feature_channels=feature_channels,
            attention_channels=attention_channels,
            num_heads=8,
            use_spectral_norm=use_spectral_norm
        )
        
        # Final output convolution for structure map generation
        self.output = nn.Sequential(
            self.norm_layer(nn.Conv2d(feature_channels * 2, attention_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(attention_channels),
            self.activation,
            self.norm_layer(nn.Conv2d(attention_channels, attention_channels // 2, kernel_size=3, padding=1)),
            nn.Tanh()  # Normalize outputs to [-1, 1] range
        )
        
        # Initialize weights for better convergence
        self._init_weights()
        
        # Move to specified device if provided
        if 'device' in kwargs:
            self.to(kwargs['device'])
    
    def _init_weights(self):
        """
        Initialize model weights for better training dynamics.
        
        Uses Kaiming initialization for convolutional layers to ensure
        proper gradient flow in deep networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Initialize biases to zero
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                # Initialize normalization parameters
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Generate structural information for masked regions.
        
        Args:
            features: Input feature tensor [B, C, H, W]
            masks: Binary masks indicating regions to inpaint [B, 1, H, W]
                  1 = area to inpaint (hole), 0 = known region
            guidance: Optional guidance information (e.g., edges, semantic maps) [B, G, H, W]
            return_attention: Whether to return attention maps for visualization
            **kwargs: Additional arguments to pass to submodules
        
        Returns:
            If return_attention is False:
                Structural feature maps [B, attention_channels//2, H, W]
            Otherwise:
                Tuple of:
                - Structural feature maps [B, attention_channels//2, H, W]
                - Dictionary of attention maps for visualization
        """
        # Ensure inputs are on the correct device
        if 'device' in kwargs:
            device = kwargs['device']
            features = features.to(device)
            masks = masks.to(device)
            if guidance is not None:
                guidance = guidance.to(device)
        
        batch_size, _, height, width = features.shape
        
        # Initial encoding
        x = self.encoder(features)
        
        # Store encoded features for skip connection
        encoded_features = x
        
        # Multi-scale processing
        x = self.downsample(x)
        
        # Apply global context module for large receptive field
        x = self.global_context(x)
        
        # Upsample back to original resolution
        x = self.upsample(x)
        
        # Apply attention mechanism focusing on valid (unmasked) regions
        # This allows the model to draw structure from existing content
        if return_attention:
            attended_features, attention_maps = self.attention(
                x, 
                masks, 
                return_attention=True
            )
        else:
            attended_features = self.attention(x, masks)
        
        # Combine attended features with skip connection from encoder
        combined_features = torch.cat([attended_features, encoded_features], dim=1)
        
        # Generate the final structural representation
        structure_features = self.output(combined_features)
        
        # Return with attention maps if requested
        if return_attention:
            return structure_features, attention_maps
        else:
            return structure_features
    
    def build_structure(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Convenient wrapper method for generating structural information.
        
        This method provides a simpler interface to the forward method,
        always returning just the structural features without attention maps.
        
        Args:
            features: Input feature tensor [B, C, H, W]
            masks: Binary masks indicating regions to inpaint [B, 1, H, W]
            guidance: Optional guidance information [B, G, H, W]
            **kwargs: Additional arguments to pass to forward method
            
        Returns:
            Structural feature maps [B, attention_channels//2, H, W]
        """
        return self.forward(features, masks, guidance, return_attention=False, **kwargs)


class ContextualAttention(nn.Module):
    """
    Contextual Attention module for structure-aware feature transformation.
    
    This module implements a patch-based attention mechanism that allows the model
    to borrow features from known regions to inform the generation of features in
    masked regions. It works by:
    
    1. Extracting patches from the unmasked regions
    2. Computing similarity between masked region features and these patches
    3. Using the similarities as attention weights to sample from known regions
    4. Reconstructing features for masked regions based on these samples
    
    This approach enables structure propagation from known to unknown regions
    while maintaining coherence and consistency.
    """
    
    def __init__(
        self,
        feature_channels: int,
        attention_channels: int = 128,
        kernel_size: int = 3,
        stride: int = 1,
        rate: int = 2,
        num_heads: int = 8,
        use_spectral_norm: bool = True
    ):
        """
        Initialize the contextual attention module.
        
        Args:
            feature_channels: Number of input feature channels
            attention_channels: Number of channels in attention computation
            kernel_size: Size of patches to extract (typically 3x3)
            stride: Stride for patch extraction
            rate: Dilation rate for patch extraction
            num_heads: Number of attention heads
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()
        
        self.feature_channels = feature_channels
        self.attention_channels = attention_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.num_heads = num_heads
        
        # Define normalization function
        norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Feature transformation for query (masked regions)
        self.query_conv = nn.Sequential(
            norm_layer(nn.Conv2d(feature_channels, attention_channels, kernel_size=1)),
            nn.InstanceNorm2d(attention_channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature transformation for key (unmasked regions)
        self.key_conv = nn.Sequential(
            norm_layer(nn.Conv2d(feature_channels, attention_channels, kernel_size=1)),
            nn.InstanceNorm2d(attention_channels),
            nn.ReLU(inplace=True)
        )
        
        # Patch fusion convolution
        self.output_conv = nn.Sequential(
            norm_layer(nn.Conv2d(feature_channels, feature_channels, kernel_size=1)),
            nn.InstanceNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
    
    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from input features for attention computation.
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            Extracted patches tensor [B, C*k*k, H_p, W_p] where
            k is kernel_size and H_p, W_p are the spatial dimensions
            after patch extraction
        """
        batch_size, channels, height, width = x.shape
        
        # Extract patches using unfold operation
        # This reshapes the tensor so each patch becomes a feature vector
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.rate,
            padding=self.rate*(self.kernel_size-1)//2,
            stride=self.stride
        )
        
        # Calculate output dimensions
        out_h = (height + 2 * (self.rate*(self.kernel_size-1)//2) - self.rate * (self.kernel_size - 1) - 1) // self.stride + 1
        out_w = (width + 2 * (self.rate*(self.kernel_size-1)//2) - self.rate * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # Reshape to [B, C*k*k, H_p*W_p]
        patches = patches.view(batch_size, channels * self.kernel_size * self.kernel_size, out_h * out_w)
        
        # Reshape to patch grid format [B, C*k*k, H_p, W_p]
        patches = patches.view(batch_size, channels * self.kernel_size * self.kernel_size, out_h, out_w)
        
        return patches
    
    def forward(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Apply contextual attention to features based on valid (unmasked) regions.
        
        Args:
            features: Input feature tensor [B, C, H, W]
            masks: Binary masks indicating regions to inpaint [B, 1, H, W]
                  1 = area to inpaint (hole), 0 = known region
            return_attention: Whether to return attention maps for visualization
            
        Returns:
            If return_attention is False:
                Attended feature tensor [B, C, H, W]
            Otherwise:
                Tuple containing:
                - Attended feature tensor [B, C, H, W]
                - Dictionary with attention maps for visualization
        """
        batch_size, channels, height, width = features.shape
        
        # Ensure masks is binary (0 or 1)
        binary_masks = (masks > 0.5).float()
        
        # Split features into masked and unmasked regions
        masked_features = features * binary_masks
        unmasked_features = features * (1 - binary_masks)
        
        # Transform features for attention computation
        q = self.query_conv(masked_features)   # Query: masked regions
        k = self.key_conv(unmasked_features)   # Key: unmasked regions
        
        # Split into multiple heads
        head_channels = self.attention_channels // self.num_heads
        q = q.view(batch_size, self.num_heads, head_channels, height, width)
        k = k.view(batch_size, self.num_heads, head_channels, height, width)
        
        # Extract patches from unmasked regions (keys)
        k_patches = []
        for h in range(self.num_heads):
            k_h = self.extract_patches(k[:, h])
            k_patches.append(k_h)
        
        # Compute attention maps between masked features and unmasked patches
        attention_maps = []
        attended_features = []
        
        for h in range(self.num_heads):
            # Get query features for this head
            q_h = q[:, h]  # [B, C/heads, H, W]
            
            # Reshape for matrix multiplication
            q_flat = q_h.view(batch_size, head_channels, -1).permute(0, 2, 1)  # [B, H*W, C/heads]
            
            # Get key patches for this head
            k_h_patches = k_patches[h]  # [B, C/heads*k*k, H_p, W_p]
            
            # Reshape key patches for matrix multiplication
            k_h_flat = k_h_patches.view(batch_size, head_channels * self.kernel_size * self.kernel_size, -1)
            k_h_flat = k_h_flat.permute(0, 2, 1)  # [B, H_p*W_p, C/heads*k*k]
            
            # Compute attention scores
            attn = torch.bmm(q_flat, k_h_flat)  # [B, H*W, H_p*W_p]
            
            # Apply softmax to get attention weights
            attn = F.softmax(attn, dim=2)
            
            # Store attention map for return if needed
            if return_attention:
                attention_maps.append(attn)
            
            # Extract patches for value (same as unmasked features)
            v_patches = self.extract_patches(unmasked_features)
            
            # Reshape for matrix multiplication
            v_flat = v_patches.view(batch_size, channels * self.kernel_size * self.kernel_size, -1)
            v_flat = v_flat.permute(0, 2, 1)  # [B, H_p*W_p, C*k*k]
            
            # Apply attention weights to sample patches
            sampled = torch.bmm(attn, v_flat)  # [B, H*W, C*k*k]
            
            # Reshape back to spatial feature map
            sampled = sampled.permute(0, 2, 1).view(batch_size, channels * self.kernel_size * self.kernel_size, height, width)
            
            # Process sampled features with a convolution to get attended features
            # Here we use a 1x1 convolution to combine the patch dimensions
            attended_h = F.conv2d(
                sampled,
                torch.ones(channels, 1, self.kernel_size, self.kernel_size).to(sampled.device),
                padding=0,
                groups=channels
            ) / (self.kernel_size * self.kernel_size)
            
            attended_features.append(attended_h)
        
        # Combine attended features from all heads
        attended = torch.cat(attended_features, dim=1)
        
        # Final processing
        output = self.output_conv(attended)
        
        # Return with attention maps if requested
        if return_attention:
            # Process attention maps for visualization
            # Average attention across heads and reshape to spatial dimensions
            attn_avg = torch.stack(attention_maps).mean(dim=0)  # [B, H*W, H_p*W_p]
            
            # Reshape to spatial dimensions for visualization
            h_p = w_p = int(attn_avg.shape[2] ** 0.5)
            attn_spatial = attn_avg.view(batch_size, height, width, h_p, w_p)
            
            # Create heatmap by summing across patch dimensions
            heatmap = attn_spatial.sum(dim=(3, 4))  # [B, H, W]
            
            # Normalize for visualization
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            return output, {
                "attention_maps": attn_avg,
                "attention_heatmap": heatmap_norm.unsqueeze(1)  # Add channel dimension
            }
        
        return output 