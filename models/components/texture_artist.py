"""
Texture Artist Module

This module provides functionality for generating detailed textures in inpainted regions
based on structural information. The TextureArtist is responsible for adding realistic
textures to the structure foundation provided by the StructureBuilder component.

The module uses a sophisticated architecture combining:
- Multi-resolution processing
- Attention mechanisms
- Style modulation
- Progressive refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

class TextureArtist(nn.Module):
    """
    TextureArtist generates realistic textures for inpainted regions.
    
    This component is the final stage in the inpainting process, responsible for:
    
    1. Adding fine-grained texture details to structural information
    2. Ensuring texture consistency with surrounding regions
    3. Maintaining global style coherence
    4. Producing high-quality, visually plausible completions
    
    The artist uses multiple scales of processing with skip connections and
    attention mechanisms to generate textures that blend seamlessly with the
    existing image content.
    """
    
    def __init__(
        self,
        in_channels: int,
        structure_channels: int,
        feature_channels: int = 64,
        output_channels: int = 3,
        num_residual_blocks: int = 8,
        num_downsamples: int = 2,
        attention_layers: int = 3,
        use_spectral_norm: bool = True,
        use_style_modulation: bool = True,
        **kwargs
    ):
        """
        Initialize the texture artist.
        
        Args:
            in_channels: Number of input channels from mask, image features
            structure_channels: Number of channels in structure features
            feature_channels: Base number of channels in internal representations
            output_channels: Number of output channels (typically 3 for RGB)
            num_residual_blocks: Number of residual blocks for refinement
            num_downsamples: Number of downsampling operations for multi-scale processing
            attention_layers: Number of attention layers for global context
            use_spectral_norm: Whether to use spectral normalization for training stability
            use_style_modulation: Whether to use style modulation for texture control
            **kwargs: Additional arguments including 'device' specification
        """
        super().__init__()
        
        # Store configuration parameters
        self.in_channels = in_channels
        self.structure_channels = structure_channels
        self.feature_channels = feature_channels
        self.output_channels = output_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_downsamples = num_downsamples
        self.use_style_modulation = use_style_modulation
        self.use_spectral_norm = use_spectral_norm
        
        # Set up normalization function based on configuration
        self.norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Input processing to merge image features, masks, and structural information
        self.input_conv = nn.Sequential(
            self.norm_layer(nn.Conv2d(in_channels + structure_channels, feature_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            self.norm_layer(nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks for multi-scale processing
        self.encoder_blocks = nn.ModuleList()
        curr_channels = feature_channels
        
        for i in range(num_downsamples):
            # Double the number of channels with each downsampling
            next_channels = curr_channels * 2
            
            # Create encoder block with downsampling
            encoder_block = nn.Sequential(
                self.norm_layer(nn.Conv2d(curr_channels, next_channels, kernel_size=4, stride=2, padding=1)),
                nn.InstanceNorm2d(next_channels),
                nn.ReLU(inplace=True),
                *[ResidualBlock(next_channels, use_spectral_norm) for _ in range(2)]
            )
            
            self.encoder_blocks.append(encoder_block)
            curr_channels = next_channels
        
        # Bottleneck blocks for global processing
        self.bottleneck = nn.Sequential(*[
            ResidualBlock(curr_channels, use_spectral_norm) 
            for _ in range(num_residual_blocks // 2)
        ])
        
        # Global attention for capturing distant dependencies
        self.global_attention = GlobalAttention(
            curr_channels, 
            heads=8,
            use_spectral_norm=use_spectral_norm
        )
        
        # Decoder blocks for upsampling back to original resolution
        self.decoder_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for i in range(num_downsamples):
            # Next channel dimension after upsampling
            next_channels = curr_channels // 2
            
            # Skip connection processor to adapt encoder features
            self.skip_connections.append(nn.Sequential(
                self.norm_layer(nn.Conv2d(curr_channels, next_channels, kernel_size=1)),
                nn.InstanceNorm2d(next_channels),
                nn.ReLU(inplace=True)
            ))
            
            # Decoder block with upsampling
            decoder_block = nn.Sequential(
                self.norm_layer(nn.ConvTranspose2d(curr_channels, next_channels, kernel_size=4, stride=2, padding=1)),
                nn.InstanceNorm2d(next_channels),
                nn.ReLU(inplace=True),
                *[ResidualBlock(next_channels, use_spectral_norm) for _ in range(2)]
            )
            
            self.decoder_blocks.append(decoder_block)
            curr_channels = next_channels
        
        # Style modulation components if enabled
        if use_style_modulation:
            # Style encoder processes reference style features
            self.style_encoder = StyleEncoder(
                feature_channels, 
                feature_channels * 4, 
                use_spectral_norm
            )
            
            # Style modulation applies style information to content features
            self.style_modulator = StyleModulator(
                feature_channels,
                feature_channels * 4
            )
        
        # Refinement blocks for final texture details
        self.refinement = nn.Sequential(*[
            ResidualBlock(feature_channels, use_spectral_norm) 
            for _ in range(num_residual_blocks // 2)
        ])
        
        # Output layers to generate final RGB image
        self.output_layers = nn.Sequential(
            self.norm_layer(nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Normalize output to [-1, 1] range
        )
        
        # Initialize weights for better convergence
        self._init_weights()
        
        # Move to specified device if provided
        if 'device' in kwargs:
            self.to(kwargs['device'])
    
    def _init_weights(self):
        """
        Initialize model weights for better training dynamics.
        
        Uses Kaiming initialization for convolutional layers and
        zero initialization for biases.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Kaiming initialization 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        features: torch.Tensor,
        structure_features: torch.Tensor,
        masks: torch.Tensor,
        style_features: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate realistic textures based on structure and style information.
        
        Args:
            features: Input feature tensor [B, C, H, W]
                     Usually masked or partial features
            structure_features: Structural information [B, structure_channels, H, W]
                              From the StructureBuilder component
            masks: Binary masks indicating regions to inpaint [B, 1, H, W]
                  1 = area to inpaint (hole), 0 = known region
            style_features: Optional style features for style control [B, style_dims, H, W]
                          Controls texture appearance, if None, extracted from input
            return_intermediate: Whether to return intermediate results
            **kwargs: Additional arguments to pass to submodules
        
        Returns:
            If return_intermediate is False:
                Inpainted RGB image with realistic textures [B, output_channels, H, W]
            Otherwise:
                Dictionary containing:
                - 'output': Final inpainted image
                - 'multi_scale_features': List of multi-scale features
                - 'attention_maps': Attention visualization if available
                - 'style_codes': Style codes if style modulation is used
        """
        # Ensure inputs are on the correct device
        if hasattr(self, 'device') and self.device:
            features = features.to(self.device)
            structure_features = structure_features.to(self.device)
            masks = masks.to(self.device)
            if style_features is not None:
                style_features = style_features.to(self.device)
        
        batch_size, _, height, width = features.shape
        intermediate_results = {}
        
        # Concatenate input features with structure features
        x = torch.cat([features, structure_features], dim=1)
        
        # Initial convolution to merge information
        x = self.input_conv(x)
        
        # Track encoder features for skip connections
        encoder_features = [x]
        
        # Encoder: multi-scale processing with progressive downsampling
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            encoder_features.append(x)
        
        # Store multi-scale features if requested
        if return_intermediate:
            intermediate_results['multi_scale_features'] = encoder_features.copy()
        
        # Bottleneck processing for global context
        x = self.bottleneck(x)
        
        # Apply global attention to capture long-range dependencies
        if return_intermediate:
            x, attention_maps = self.global_attention(x, return_attention=True)
            intermediate_results['attention_maps'] = attention_maps
        else:
            x = self.global_attention(x)
        
        # Process style information if enabled
        style_codes = None
        if self.use_style_modulation:
            # Extract style features from input if not provided
            if style_features is None:
                # Use unmasked regions as style reference
                style_refs = features * (1 - masks)
                style_codes = self.style_encoder(style_refs)
            else:
                style_codes = self.style_encoder(style_features)
            
            # Store style codes if requested
            if return_intermediate:
                intermediate_results['style_codes'] = style_codes
        
        # Decoder: progressive upsampling with skip connections from encoder
        for i, (decoder_block, skip_connection) in enumerate(zip(self.decoder_blocks, self.skip_connections)):
            # Get corresponding encoder features (in reverse order)
            skip_features = encoder_features[-(i+2)]
            
            # Process skip features and concatenate with current features
            processed_skip = skip_connection(skip_features)
            
            # Apply decoder block (upsampling)
            x = decoder_block(x)
            
            # Add processed skip features (residual connection)
            x = x + processed_skip
            
            # Apply style modulation at appropriate resolution if enabled
            if self.use_style_modulation and style_codes is not None and i == len(self.decoder_blocks) - 1:
                x = self.style_modulator(x, style_codes)
        
        # Apply final refinement blocks
        x = self.refinement(x)
        
        # Generate final output image
        output = self.output_layers(x)
        
        # Apply mask compositing: output * mask + original * (1-mask)
        output = output * masks + features[:, :self.output_channels] * (1 - masks)
        
        # Return intermediate results if requested
        if return_intermediate:
            intermediate_results['output'] = output
            return intermediate_results
        
        return output


class ResidualBlock(nn.Module):
    """
    Residual Block with instance normalization and spectral normalization.
    
    This block uses a residual connection to improve gradient flow and
    help maintain spatial information. Each block consists of two convolutional
    layers with normalization and ReLU activation. The input is added to the
    output via a residual connection.
    """
    
    def __init__(self, channels: int, use_spectral_norm: bool = True):
        """
        Initialize the residual block.
        
        Args:
            channels: Number of input and output channels
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()
        
        # Define normalization function
        norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Main branch
        self.block = nn.Sequential(
            norm_layer(nn.Conv2d(channels, channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            norm_layer(nn.Conv2d(channels, channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(channels)
        )
        
        # Final activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features through the residual block.
        
        Args:
            x: Input feature tensor [B, channels, H, W]
            
        Returns:
            Feature tensor with applied residual connection [B, channels, H, W]
        """
        # Apply main branch
        residual = self.block(x)
        
        # Add residual connection
        out = x + residual
        
        # Apply activation
        out = self.relu(out)
        
        return out


class GlobalAttention(nn.Module):
    """
    Multi-head self-attention module for capturing global dependencies.
    
    This module implements a spatial self-attention mechanism that allows each
    position in the feature map to attend to all other positions, effectively
    capturing long-range dependencies that are difficult to model with 
    convolutional operations.
    """
    
    def __init__(
        self,
        channels: int,
        heads: int = 8,
        use_spectral_norm: bool = True
    ):
        """
        Initialize the global attention module.
        
        Args:
            channels: Number of input and output channels
            heads: Number of attention heads
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()
        
        self.channels = channels
        self.heads = heads
        self.head_channels = channels // heads
        self.scale = self.head_channels ** -0.5  # Scaling factor for dot products
        
        # Define normalization function
        norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Query, key, value projections
        self.query_conv = norm_layer(nn.Conv2d(channels, channels, kernel_size=1))
        self.key_conv = norm_layer(nn.Conv2d(channels, channels, kernel_size=1))
        self.value_conv = norm_layer(nn.Conv2d(channels, channels, kernel_size=1))
        
        # Output projection
        self.output_conv = norm_layer(nn.Conv2d(channels, channels, kernel_size=1))
        
        # Layer normalization for stability
        self.norm = nn.GroupNorm(groups=heads, num_channels=channels)
        
        # Gamma parameter for residual connection scaling
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Apply self-attention to input features.
        
        Args:
            x: Input feature tensor [B, C, H, W]
            return_attention: Whether to return attention maps for visualization
            
        Returns:
            If return_attention is False:
                Processed feature tensor with global context [B, C, H, W]
            Otherwise:
                Tuple containing:
                - Processed feature tensor
                - Dictionary with attention maps for visualization
        """
        batch_size, _, height, width = x.shape
        
        # Apply normalization
        x_norm = self.norm(x)
        
        # Generate query, key, value projections
        q = self.query_conv(x_norm)  # [B, C, H, W]
        k = self.key_conv(x_norm)    # [B, C, H, W]
        v = self.value_conv(x_norm)  # [B, C, H, W]
        
        # Reshape to separate the heads and flatten spatial dimensions
        q = q.reshape(batch_size, self.heads, self.head_channels, -1)  # [B, H, C/H, HW]
        k = k.reshape(batch_size, self.heads, self.head_channels, -1)  # [B, H, C/H, HW]
        v = v.reshape(batch_size, self.heads, self.head_channels, -1)  # [B, H, C/H, HW]
        
        # Transpose for matrix multiplication
        q = q.permute(0, 1, 3, 2)  # [B, H, HW, C/H]
        
        # Calculate attention scores
        attn = torch.matmul(q, k)  # [B, H, HW, HW]
        
        # Apply scaling
        attn = attn * self.scale
        
        # Normalize with softmax
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v.permute(0, 1, 3, 2))  # [B, H, HW, C/H]
        
        # Reshape back to original spatial dimensions
        out = out.permute(0, 1, 3, 2).reshape(batch_size, self.channels, height, width)
        
        # Apply output projection
        out = self.output_conv(out)
        
        # Apply residual connection with learnable weighting
        out = x + self.gamma * out
        
        if return_attention:
            # Return attention maps for visualization
            attn_maps = {'attention': attn.detach()}
            return out, attn_maps
        
        return out


class StyleEncoder(nn.Module):
    """
    Style Encoder extracts style information from reference images.
    
    This module encodes style features that can be used to control the
    texture appearance in the generated output. It progressively reduces
    spatial dimensions while increasing feature depth to capture style
    characteristics at multiple scales.
    """
    
    def __init__(
        self,
        in_channels: int,
        style_channels: int,
        use_spectral_norm: bool = True
    ):
        """
        Initialize the style encoder.
        
        Args:
            in_channels: Number of input channels
            style_channels: Number of style feature channels to generate
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.style_channels = style_channels
        
        # Define normalization function
        norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Progressive downsampling to extract hierarchical style features
        self.encoder = nn.Sequential(
            # Initial layer maintains spatial dimensions
            norm_layer(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # First downsample layer: in_channels -> in_channels*2
            norm_layer(nn.Conv2d(in_channels, in_channels*2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
            
            # Second downsample layer: in_channels*2 -> in_channels*4
            norm_layer(nn.Conv2d(in_channels*2, in_channels*4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(in_channels*4),
            nn.ReLU(inplace=True),
            
            # Third downsample layer: in_channels*4 -> style_channels
            norm_layer(nn.Conv2d(in_channels*4, style_channels, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(style_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract style features from input.
        
        Args:
            x: Input feature tensor [B, in_channels, H, W]
            
        Returns:
            Style feature tensor [B, style_channels, H/8, W/8]
        """
        # Apply encoder to extract hierarchical style features
        style_features = self.encoder(x)
        
        return style_features


class StyleModulator(nn.Module):
    """
    Style Modulator applies style information to content features.
    
    This module performs adaptive instance normalization (AdaIN) to
    transfer style characteristics from one set of features to another.
    It recalibrates the mean and variance of content features to match
    those of style features, effectively transferring style.
    """
    
    def __init__(
        self,
        in_channels: int,
        style_channels: int
    ):
        """
        Initialize the style modulator.
        
        Args:
            in_channels: Number of content feature channels
            style_channels: Number of style feature channels
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.style_channels = style_channels
        
        # Style adaptation layers to generate scale and bias parameters
        self.style_transform = nn.Sequential(
            nn.Conv2d(style_channels, style_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global pooling to get style codes
            nn.Flatten(),
            nn.Linear(style_channels, in_channels * 2)  # For scale and bias
        )
        
        # Initialize the bias of the last linear layer to 0 and 1
        # This makes the initial modulation close to identity
        self.style_transform[-1].bias.data[:in_channels] = 1.0
        self.style_transform[-1].bias.data[in_channels:] = 0.0
    
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Apply style modulation to content features.
        
        Args:
            content: Content feature tensor [B, in_channels, H, W]
            style: Style feature tensor [B, style_channels, H_s, W_s]
                 (spatial dimensions can differ from content)
            
        Returns:
            Style-modulated content features [B, in_channels, H, W]
        """
        batch_size, _, height, width = content.shape
        
        # Compute instance normalization statistics
        mean = content.mean(dim=(2, 3), keepdim=True)
        std = content.std(dim=(2, 3), keepdim=True) + 1e-5
        
        # Normalize content features
        normalized = (content - mean) / std
        
        # Generate scale and bias from style features
        style_params = self.style_transform(style)
        style_params = style_params.view(batch_size, self.in_channels * 2, 1, 1)
        
        # Extract scale and bias
        scale = style_params[:, :self.in_channels]
        bias = style_params[:, self.in_channels:]
        
        # Apply modulation
        modulated = normalized * scale + bias
        
        return modulated 