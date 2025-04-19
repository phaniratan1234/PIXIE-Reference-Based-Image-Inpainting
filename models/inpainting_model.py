"""
Inpainting Model Module

This module defines a comprehensive architecture for image inpainting, combining multiple
specialized components to create a robust and flexible inpainting system that can handle
diverse image contexts and challenging inpainting scenarios.

The model employs a pipeline approach integrating:
- Object detection to identify and preserve important scene elements
- Scene classification to understand global context
- Structural analysis to capture underlying image structure
- Uncertainty estimation to quantify prediction confidence
- Structure building to create coherent structural foundations
- Texture artistry to generate detailed and realistic textures

Each component can be enabled/disabled via configuration, allowing for flexible model setups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any, Union

from models.components.detection import ObjectDetector
from models.components.scene_classifier import SceneClassifier
from models.components.structural_analyzer import StructuralAnalyzer
from models.components.uncertainty_estimator import UncertaintyEstimator
from models.components.structure_builder import StructureBuilder
from models.components.texture_artist import TextureArtist

class InpaintingModel(nn.Module):
    """
    Multi-component inpainting architecture that combines specialized sub-networks for
    comprehensive image inpainting capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the inpainting model with configurable components.
        
        Args:
            config: Dictionary containing configuration parameters for the entire model
                  Must include the following keys:
                  - 'device': Device to run the model on (cuda, cpu, mps)
                  - 'feature_channels': Number of channels in feature maps
                  - 'components': Dict with configuration for each component
                    Each component config should have 'enabled' flag and component-specific parameters
        """
        super().__init__()
        
        # Store configuration for reference
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_channels = config.get('feature_channels', 64)
        
        # Initialize components based on configuration
        components_config = config.get('components', {})
        
        # Object detection component for identifying and preserving important scene elements
        detection_config = components_config.get('detection', {'enabled': False})
        self.use_detection = detection_config.get('enabled', False)
        if self.use_detection:
            self.object_detector = ObjectDetector(
                in_channels=3,  # RGB input
                **detection_config,
                device=self.device
            )
        
        # Scene classification component for understanding global context
        scene_config = components_config.get('scene_classification', {'enabled': False})
        self.use_scene_classification = scene_config.get('enabled', False)
        if self.use_scene_classification:
            self.scene_classifier = SceneClassifier(
                in_channels=3,  # RGB input
                **scene_config,
                device=self.device
            )
        
        # Structural analysis component for capturing underlying image structure
        structure_analysis_config = components_config.get('structural_analysis', {'enabled': True})
        self.use_structural_analysis = structure_analysis_config.get('enabled', True)
        if self.use_structural_analysis:
            self.structural_analyzer = StructuralAnalyzer(
                in_channels=3,  # RGB input
                **structure_analysis_config,
                device=self.device
            )
        
        # Uncertainty estimation component for quantifying prediction confidence
        uncertainty_config = components_config.get('uncertainty_estimation', {'enabled': True})
        self.use_uncertainty = uncertainty_config.get('enabled', True)
        if self.use_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(
                in_channels=self.feature_channels,
                **uncertainty_config,
                device=self.device
            )
        
        # Structure building component for creating coherent structural foundations
        structure_config = components_config.get('structure_building', {'enabled': True})
        self.use_structure_building = structure_config.get('enabled', True)
        if self.use_structure_building:
            self.structure_builder = StructureBuilder(
                in_channels=self.feature_channels,
                **structure_config,
                device=self.device
            )
        
        # Texture artistry component for generating detailed and realistic textures
        texture_config = components_config.get('texture_artistry', {'enabled': True})
        self.use_texture_artistry = texture_config.get('enabled', True)
        if self.use_texture_artistry:
            self.texture_artist = TextureArtist(
                in_channels=self.feature_channels,
                **texture_config,
                device=self.device
            )
        
        # Store structure/texture channel dimensions for reference during forward pass
        # These will be used to ensure channel compatibility during processing
        self.structure_out_channels = self.structure_builder.out_channels
        self.texture_in_channels = self.texture_artist.in_channels
        
        # Create a channel adapter if needed (created during forward pass when device is known)
        # This will handle channel dimension mismatches between structure_builder and texture_artist
        self.channel_adapter = None
        
        # Configuration flags for enabling/disabling specific components
        self.use_classification = self.use_scene_classification
        self.skip_uncertainty = not self.use_uncertainty
    
    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through the inpainting model pipeline.
        
        Processing flow:
        1. Initial component analysis (detection, scene classification, structural analysis)
        2. Structure generation based on analysis results
        3. Texture refinement using structural guides
        4. Uncertainty estimation for quality assessment
        5. Final composition of inpainted result
        
        Args:
            images: Input images with shape [B, 3, H, W]
                  Values should be normalized to range [0, 1]
            masks: Binary masks indicating regions to inpaint with shape [B, 1, H, W]
                  1 = area to inpaint (hole), 0 = known region
            return_intermediates: Whether to return intermediate results (default: False)
                               Useful for debugging and visualization
        
        Returns:
            If return_intermediates is False:
                inpainted_images: The final inpainted images with shape [B, 3, H, W]
            
            If return_intermediates is True:
                (inpainted_images, intermediates): Tuple containing the final inpainted images
                                                and a dictionary with intermediate results
        """
        batch_size, _, height, width = images.shape
        
        # Store intermediates if requested
        intermediates = {} if return_intermediates else None
        
        # Apply object detection if enabled
        objects_info = None
        if self.use_detection:
            objects_info = self.object_detector(images)
            if return_intermediates:
                intermediates['objects_info'] = objects_info
        
        # Apply scene classification if enabled
        scene_features = None
        if self.use_scene_classification:
            scene_features = self.scene_classifier(images)
            if return_intermediates:
                intermediates['scene_features'] = scene_features
        
        # Apply structural analysis
        structural_features = None
        if self.use_structural_analysis:
            # Generate structural representation of the input image
            structural_features = self.structural_analyzer(images, masks)
            if return_intermediates:
                intermediates['structural_features'] = structural_features
        
        # Build structure for inpainting
        structure = None
        if self.use_structure_building and structural_features is not None:
            # Generate structural content for the masked regions
            structure = self.structure_builder(
                structural_features,
                masks,
                objects_info=objects_info,
                scene_features=scene_features
            )
            if return_intermediates:
                intermediates['structure'] = structure
        
        # Apply texture artistry
        inpainted_images = None
        if self.use_texture_artistry:
            # Generate detailed texture based on structural guidance
            inpainted_images = self.texture_artist(
                images,
                masks,
                structure=structure,
                objects_info=objects_info,
                scene_features=scene_features
            )
            if return_intermediates:
                intermediates['inpainted_images_pre_uncertainty'] = inpainted_images.clone()
        
        # Apply uncertainty estimation if enabled
        uncertainty_maps = None
        if self.use_uncertainty and structural_features is not None:
            # Estimate uncertainty in the generated content
            uncertainty_result = self.uncertainty_estimator(
                structural_features,
                masks=masks,
                structure=structure
            )
            
            # Extract uncertainty maps
            uncertainty_maps = uncertainty_result.get('uncertainty_maps', None)
            confidence_maps = uncertainty_result.get('confidence_maps', None)
            
            if return_intermediates:
                intermediates['uncertainty_maps'] = uncertainty_maps
                intermediates['confidence_maps'] = confidence_maps
            
            # Apply confidence-weighted blending if available
            if inpainted_images is not None and confidence_maps is not None:
                # Use confidence maps to refine the inpainted result
                texture_confidence = confidence_maps.get('texture', None)
                if texture_confidence is not None:
                    # Create inverted mask (1 for known regions, 0 for holes)
                    inv_masks = 1 - masks
                    
                    # Weight the inpainted regions by confidence
                    weighted_inpaint = inpainted_images * texture_confidence
                    
                    # Combine original (known) regions with inpainted (unknown) regions
                    inpainted_images = images * inv_masks + weighted_inpaint * masks
        
        # If no texture artist, create a basic inpainted result using structure
        if inpainted_images is None and structure is not None:
            # Use structure as basic inpainting result
            inpainted_images = images * (1 - masks) + structure * masks
        
        # Return the requested outputs
        if return_intermediates:
            return inpainted_images, intermediates
        else:
            return inpainted_images
    
    def _compute_losses(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        inpainted_images: torch.Tensor,
        intermediates: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute various loss terms for training the inpainting model.
        
        This method calculates multiple loss components that guide the learning process:
        1. Reconstruction loss: Measures pixel-level accuracy in the inpainted region
        2. Perceptual loss: Ensures perceptual similarity using feature-level comparison
        3. Style loss: Enforces style consistency between inpainted and original regions
        4. Structure loss: Maintains structural coherence in the inpainted area
        5. Adversarial loss: Improves realism if adversarial training is enabled
        
        Each loss component can be weighted according to the model configuration to
        balance their relative importance during training.
        
        Args:
            images: Original input images [B, 3, H, W]
            masks: Binary masks indicating regions to inpaint [B, 1, H, W]
                  1 = area to inpaint (hole), 0 = known region
            inpainted_images: Generated inpainted images [B, 3, H, W]
            intermediates: Dictionary containing intermediate results from forward pass
                          Used for computing specialized loss terms
        
        Returns:
            Dictionary mapping loss names to their computed tensor values.
            The 'total_loss' key contains the weighted sum of all individual losses.
        """
        # Dictionary to store all loss components
        losses = {}
        
        # Pixel-wise reconstruction loss (L1)
        # Focuses on the inpainted region to ensure pixel-level accuracy
        l1_loss = F.l1_loss(
            inpainted_images * masks,  # Only consider inpainted regions
            images * masks
        )
        losses['reconstruction_loss'] = l1_loss
        
        # Initialize total loss with weighted reconstruction loss
        l1_weight = self.config.get('loss_weights', {}).get('reconstruction', 1.0)
        total_loss = l1_weight * l1_loss
        
        # Perceptual loss (if VGG features are available)
        if 'vgg_features' in intermediates:
            vgg_weight = self.config.get('loss_weights', {}).get('perceptual', 0.1)
            vgg_real = intermediates['vgg_features']['real']
            vgg_fake = intermediates['vgg_features']['fake']
            
            perceptual_loss = 0
            # Compute MSE loss for each feature layer
            for real_feat, fake_feat in zip(vgg_real, vgg_fake):
                perceptual_loss += F.mse_loss(real_feat, fake_feat)
            
            losses['perceptual_loss'] = perceptual_loss
            total_loss += vgg_weight * perceptual_loss
        
        # Style loss for texture consistency
        if 'texture_features' in intermediates:
            style_weight = self.config.get('loss_weights', {}).get('style', 0.5)
            texture_real = intermediates['texture_features']['real']
            texture_fake = intermediates['texture_features']['fake']
            
            style_loss = 0
            # Compute Gram matrix similarity for style matching
            for real_feat, fake_feat in zip(texture_real, texture_fake):
                # Reshape for Gram matrix computation
                b, c, h, w = real_feat.shape
                real_gram = self._gram_matrix(real_feat.reshape(b, c, -1))
                fake_gram = self._gram_matrix(fake_feat.reshape(b, c, -1))
                
                style_loss += F.mse_loss(real_gram, fake_gram)
            
            losses['style_loss'] = style_loss
            total_loss += style_weight * style_loss
        
        # Structure coherence loss
        if 'structure_features' in intermediates and self.use_structural_analysis:
            structure_weight = self.config.get('loss_weights', {}).get('structure', 0.3)
            struct_real = intermediates['structure_features']['edges']
            struct_fake = intermediates['structure_features']['inpainted_edges']
            
            # Edge consistency in the inpainted region
            structure_loss = F.mse_loss(
                struct_fake * masks,
                struct_real * masks
            )
            
            losses['structure_loss'] = structure_loss
            total_loss += structure_weight * structure_loss
        
        # Adversarial loss (if discriminator is used)
        if 'disc_outputs' in intermediates:
            adv_weight = self.config.get('loss_weights', {}).get('adversarial', 0.1)
            disc_real = intermediates['disc_outputs']['real']
            disc_fake = intermediates['disc_outputs']['fake']
            
            # Non-saturating GAN loss for the generator
            adv_loss = -torch.mean(torch.log(disc_fake + 1e-8))
            
            losses['adversarial_loss'] = adv_loss
            total_loss += adv_weight * adv_loss
        
        # Total variation loss for smoothness
        tv_weight = self.config.get('loss_weights', {}).get('total_variation', 0.01)
        if tv_weight > 0:
            # Compute gradients
            tv_h = torch.mean(torch.abs(inpainted_images[:, :, :-1, :] - inpainted_images[:, :, 1:, :]))
            tv_w = torch.mean(torch.abs(inpainted_images[:, :, :, :-1] - inpainted_images[:, :, :, 1:]))
            tv_loss = tv_h + tv_w
            
            losses['tv_loss'] = tv_loss
            total_loss += tv_weight * tv_loss
        
        # Store total loss
        losses['total_loss'] = total_loss
        
        return losses
    
    def _gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix for style loss calculation.
        
        The Gram matrix captures style information by computing the correlation
        between different feature channels. It's used in style loss computation
        to match texture characteristics.
        
        Args:
            features: Feature tensor [B, C, N] where N is the number of spatial locations
            
        Returns:
            Gram matrix [B, C, C] representing channel-wise correlations
        """
        batch_size, channels, num_locations = features.shape
        features_t = features.transpose(1, 2)  # [B, N, C]
        gram = torch.bmm(features, features_t)  # [B, C, C]
        
        # Normalize by the number of spatial locations
        return gram / num_locations
    
    def inference(
        self,
        images: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified interface for performing inference with the inpainting model.
        
        This method provides a straightforward way to generate inpainted images without
        returning intermediate results. It handles the necessary preprocessing and
        ensures the inputs are on the correct device.
        
        Use this method for:
        - Production inference
        - Batch processing of images
        - When intermediate results are not needed
        
        For debugging, visualization, or detailed analysis, use the forward method
        with return_intermediates=True instead.
        
        Args:
            images: Input images with shape [B, 3, H, W]
                  Values should be normalized to range [0, 1]
            masks: Binary masks indicating regions to inpaint with shape [B, 1, H, W]
                  1 = area to inpaint (hole), 0 = known region
        
        Returns:
            inpainted_images: The final inpainted images with shape [B, 3, H, W]
        """
        # Ensure inputs are on the correct device
        if hasattr(self, 'device') and self.device:
            images = images.to(self.device)
            masks = masks.to(self.device)
        
        # Set model to evaluation mode
        self.eval()
        
        # Disable gradient computation for inference
        with torch.no_grad():
            # Process images through the model
            inpainted_images = self.forward(images, masks, return_intermediates=False)
        
        return inpainted_images 