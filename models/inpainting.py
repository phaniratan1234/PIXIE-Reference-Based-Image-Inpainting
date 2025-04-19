#!/usr/bin/env python3
"""
Complete image inpainting model that integrates all components.
Coordinates the full inpainting pipeline from masked input to final result.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import logging

from .base import BaseModel
from .detection import YOLODetector
from .scene import SceneClassifier
from .structure import StructuralAnalysisModule
from .uncertainty import UncertaintyNetwork
from .builder import StructureBuilder
from .texture import TextureArtist

logger = logging.getLogger(__name__)

class ComprehensiveInpaintingModel(BaseModel):
    """
    Complete inpainting model integrating all components into a unified pipeline.
    
    The model follows a multi-stage approach:
    1. Scene understanding (scene classification, object detection)
    2. Structural analysis (edge detection, line detection)
    3. Uncertainty estimation
    4. Structure reconstruction
    5. Texture synthesis
    """
    
    def __init__(self, config=None):
        """
        Initialize comprehensive inpainting model.
        
        Args:
            config (dict, optional): Configuration dictionary with parameters for all components
        """
        super().__init__(config)
        
        # Default configuration values
        self.scene_enabled = self.config.get('scene_enabled', True)
        self.detection_enabled = self.config.get('detection_enabled', True)
        self.structure_enabled = self.config.get('structure_enabled', True)
        self.uncertainty_enabled = self.config.get('uncertainty_enabled', True)
        self.exemplar_guided = self.config.get('exemplar_guided', False)
        self.progressive_training = self.config.get('progressive_training', True)
        
        # Initialize components (will be built in stages)
        self.scene_classifier = None
        self.object_detector = None
        self.structural_analyzer = None
        self.uncertainty_estimator = None
        self.structure_builder = None
        self.texture_artist = None
        
        # Build all components
        self._build_components()
        
    def _build_components(self):
        """Build all model components based on configuration."""
        logger.info("Building comprehensive inpainting model components...")
        
        # Scene classifier
        if self.scene_enabled:
            logger.info("Building scene classifier...")
            scene_config = self.config.get('scene', {})
            self.scene_classifier = SceneClassifier(scene_config)
        
        # Object detector
        if self.detection_enabled:
            logger.info("Building object detector...")
            detection_config = self.config.get('detection', {})
            self.object_detector = YOLODetector(detection_config)
        
        # Structural analyzer
        if self.structure_enabled:
            logger.info("Building structural analyzer...")
            structure_config = self.config.get('structure', {})
            self.structural_analyzer = StructuralAnalysisModule(structure_config)
        
        # Uncertainty estimator
        if self.uncertainty_enabled:
            logger.info("Building uncertainty estimator...")
            uncertainty_config = self.config.get('uncertainty', {})
            self.uncertainty_estimator = UncertaintyNetwork(uncertainty_config)
        
        # Structure builder
        logger.info("Building structure builder...")
        builder_config = self.config.get('builder', {})
        self.structure_builder = StructureBuilder(builder_config)
        
        # Texture artist
        logger.info("Building texture artist...")
        texture_config = self.config.get('texture', {})
        self.texture_artist = TextureArtist(texture_config)
        
        logger.info("All inpainting model components built successfully.")
    
    def forward(self, batch):
        """
        Forward pass through the comprehensive inpainting model.
        
        Args:
            batch (dict): Input batch containing:
                - image: Original image [B, 3, H, W]
                - mask: Binary mask [B, 1, H, W]
                - masked_image: Masked image [B, 3, H, W]
                - exemplar (optional): Example image for style reference [B, 3, H, W]
                
        Returns:
            dict: Inpainting results with all intermediate outputs
        """
        results = {}
        
        # Extract inputs
        original_image = batch['image']
        mask = batch['mask']
        masked_image = batch['masked_image']
        exemplar = batch.get('exemplar', None)
        
        batch_size, _, height, width = original_image.shape
        device = original_image.device
        
        # Stage 1: Scene understanding
        scene_features = None
        semantic_map = None
        scene_masks = None
        
        if self.scene_enabled and self.scene_classifier is not None:
            logger.debug("Running scene classifier...")
            scene_results = self.scene_classifier(original_image)
            results['scene'] = scene_results
            
            # Get scene-specific attention masks
            scene_masks = self.scene_classifier.get_scene_masks(
                original_image, scene_results['scene_idx'])
            results['scene_masks'] = scene_masks
            
            # Get scene embeddings
            scene_features = self.scene_classifier.get_scene_embedding(original_image)
            
        if self.detection_enabled and self.object_detector is not None:
            logger.debug("Running object detector...")
            # Get semantic map from object detection
            semantic_map = self.object_detector.get_semantic_map(original_image)
            results['semantic_map'] = semantic_map
        
        # Stage 2: Structural analysis
        logger.debug("Running structural analysis...")
        
        # Generate empty defaults if modules are disabled
        if semantic_map is None:
            semantic_map = torch.zeros((batch_size, 1, height, width), device=device)
        
        if scene_masks is None:
            scene_masks = torch.ones((batch_size, 1, height, width), device=device)
        
        # Run structural analyzer
        structure_results = None
        if self.structure_enabled and self.structural_analyzer is not None:
            structure_results = self.structural_analyzer(
                masked_image, mask, semantic_map)
            results['structure'] = structure_results
        
        edge_map = structure_results['edge_map'] if structure_results else torch.zeros_like(mask)
        line_map = structure_results['line_map'] if structure_results else torch.zeros_like(mask)
        
        # Stage 3: Uncertainty estimation
        uncertainty_map = None
        if self.uncertainty_enabled and self.uncertainty_estimator is not None:
            logger.debug("Estimating uncertainty...")
            uncertainty_results = self.uncertainty_estimator(
                masked_image, mask, edge_map, semantic_map)
            uncertainty_map = uncertainty_results['uncertainty_map']
            results['uncertainty'] = uncertainty_results
        
        # Stage 4: Structure building
        logger.debug("Building structural framework...")
        builder_inputs = {
            'masked_image': masked_image,
            'mask': mask,
            'edge_map': edge_map,
            'line_map': line_map,
            'semantic_map': semantic_map,
            'uncertainty_map': uncertainty_map,
            'scene_features': scene_features,
        }
        
        builder_results = self.structure_builder(
            masked_image, mask, edge_map, uncertainty_map)
        results['builder'] = builder_results
        
        # Extract structure map from builder results
        structure_map = builder_results['structure_map']
        
        # Stage 5: Texture synthesis
        logger.debug("Generating textures...")
        texture_results = self.texture_artist(
            masked_image, mask, structure_map, edge_map, exemplar)
        results['texture'] = texture_results
        
        # Final output is the composite from texture artist
        results['output'] = texture_results['composite']
        
        return results
    
    def inpaint(self, image, mask, exemplar=None):
        """
        Perform inpainting on a single image.
        
        Args:
            image (torch.Tensor or np.ndarray): Input image [3, H, W] or [H, W, 3]
            mask (torch.Tensor or np.ndarray): Binary mask [1, H, W] or [H, W, 1]
            exemplar (torch.Tensor or np.ndarray, optional): Example image for style reference
            
        Returns:
            dict: Inpainting results
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(image, np.ndarray):
            # Convert HWC to CHW if needed
            if image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float() / 255.0
        
        if isinstance(mask, np.ndarray):
            # Convert HWC to CHW if needed
            if mask.shape[-1] == 1:
                mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(mask).float()
        
        if exemplar is not None and isinstance(exemplar, np.ndarray):
            # Convert HWC to CHW if needed
            if exemplar.shape[-1] == 3:
                exemplar = exemplar.transpose(2, 0, 1)
            exemplar = torch.from_numpy(exemplar).float() / 255.0
        
        # Add batch dimension if not present
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if exemplar is not None and exemplar.dim() == 3:
            exemplar = exemplar.unsqueeze(0)
        
        # Move to device
        device = next(self.parameters()).device
        image = image.to(device)
        mask = mask.to(device)
        if exemplar is not None:
            exemplar = exemplar.to(device)
        
        # Create masked image (for consistency with training)
        masked_image = image * (1 - mask)
        
        # Prepare batch
        batch = {
            'image': image,
            'mask': mask,
            'masked_image': masked_image,
        }
        
        if exemplar is not None:
            batch['exemplar'] = exemplar
            
        # Switch to eval mode
        self.eval()
        
        # Run forward pass
        with torch.no_grad():
            results = self.forward(batch)
        
        # Convert tensors to numpy arrays for easier handling
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        results[key][subkey] = subvalue.cpu().numpy()
            elif isinstance(value, torch.Tensor):
                results[key] = value.cpu().numpy()
        
        return results
    
    def train_step(self, batch, optimizer, step=0, total_steps=0):
        """
        Perform a single training step.
        
        Args:
            batch (dict): Input batch with image, mask, masked_image
            optimizer (torch.optim.Optimizer): Model optimizer
            step (int): Current step number
            total_steps (int): Total number of steps
            
        Returns:
            dict: Training statistics and losses
        """
        # Enable training mode
        self.train()
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        results = self.forward(batch)
        
        # Calculate losses
        loss_dict = self._compute_losses(batch, results, step, total_steps)
        
        # Total loss
        total_loss = sum(loss_dict.values())
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Return losses and stats
        stats = {
            'total_loss': total_loss.item(),
            **{k: v.item() for k, v in loss_dict.items()}
        }
        
        return stats
    
    def _compute_losses(self, batch, results, step=0, total_steps=0):
        """
        Compute all losses for training.
        
        Args:
            batch (dict): Input batch
            results (dict): Model outputs
            step (int): Current step
            total_steps (int): Total steps
            
        Returns:
            dict: Loss terms
        """
        # Extract ground truth
        gt_image = batch['image']
        mask = batch['mask']
        
        # Final output
        output = results['output']
        
        # Progressive training factor (gradually increase importance of structure)
        if self.progressive_training and total_steps > 0:
            progress = min(1.0, step / (total_steps * 0.5))
        else:
            progress = 1.0
            
        # Initialize losses
        losses = {}
        
        # Reconstruction loss (L1 + perceptual)
        losses['l1_loss'] = F.l1_loss(output * mask, gt_image * mask) * 10.0
        
        # Add perceptual loss if VGG is available
        if hasattr(self, 'vgg') and self.vgg is not None:
            losses['perceptual_loss'] = self._perceptual_loss(output, gt_image) * 0.1
        
        # Structure loss (if structure builder was used)
        if 'builder' in results and progress > 0:
            structure_map = results['builder']['structure_map']
            if 'structure' in results:
                gt_edge_map = results['structure']['edge_map']
                losses['structure_loss'] = F.mse_loss(
                    structure_map * mask, gt_edge_map * mask
                ) * (5.0 * progress)
        
        # Texture loss (adversarial loss)
        if hasattr(self, 'discriminator') and self.discriminator is not None:
            # Fake score
            fake_pred = self.discriminator(output)
            losses['gen_loss'] = -fake_pred.mean() * 0.1
            
            # Add GAN feature matching loss if needed
        
        return losses
    
    def _perceptual_loss(self, output, target):
        """
        Calculate perceptual loss using VGG features.
        
        Args:
            output (torch.Tensor): Model output image
            target (torch.Tensor): Ground truth image
            
        Returns:
            torch.Tensor: Perceptual loss
        """
        # This is just a placeholder - a proper implementation would use
        # a pretrained VGG network to extract features
        
        # Get VGG features
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        
        # Calculate L1 distance between features
        perceptual_loss = 0.0
        for out_feat, target_feat in zip(output_features, target_features):
            perceptual_loss += F.l1_loss(out_feat, target_feat)
            
        return perceptual_loss / len(output_features)
    
    def load(self, checkpoint_path, components=None):
        """
        Load model from checkpoint, with optional component selection.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            components (list, optional): List of component names to load
                                        If None, load all components
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if components is None:
            # Load entire model
            self.load_state_dict(checkpoint['model_state'])
            logger.info(f"Loaded full model from {checkpoint_path}")
        else:
            # Selective loading
            model_dict = self.state_dict()
            
            # Filter checkpoint state dict based on component names
            filtered_state = {}
            for k, v in checkpoint['model_state'].items():
                # Check if this parameter belongs to any of the requested components
                if any(k.startswith(comp) for comp in components):
                    filtered_state[k] = v
            
            # Update model state dict
            model_dict.update(filtered_state)
            self.load_state_dict(model_dict)
            
            logger.info(f"Loaded components {components} from {checkpoint_path}")
        
        # Return additional info from checkpoint
        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'best_score': checkpoint.get('best_score', float('inf'))
        } 