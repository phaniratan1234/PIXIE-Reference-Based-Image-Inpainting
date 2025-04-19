#!/usr/bin/env python3
"""
Scene classification module for image inpainting using a modified ResNet50.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from .base import BaseModel

class GlobalContextPooling(nn.Module):
    """
    Global context pooling module that combines features from multiple scales.
    Implements a spatial pyramid pooling approach.
    """
    
    def __init__(self, in_features, out_features, levels=(1, 2, 4)):
        """
        Initialize global context pooling.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            levels (tuple): Pooling grid sizes
        """
        super().__init__()
        
        # Calculate total number of bins across all levels
        num_bins = sum(level * level for level in levels)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Conv2d(in_features * num_bins, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
        
        self.levels = levels
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        """
        Forward pass of global context pooling.
        
        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]
            
        Returns:
            torch.Tensor: Pooled features
        """
        batch_size, in_features, h, w = x.shape
        
        # List to store pooled features
        pooled_features = []
        
        for level in self.levels:
            # Calculate kernel size and stride for adaptive pooling
            kernel_h = int(np.ceil(h / level))
            kernel_w = int(np.ceil(w / level))
            stride_h = int(np.floor(h / level))
            stride_w = int(np.floor(w / level))
            
            # Implement pooling at this level
            for i in range(level):
                for j in range(level):
                    # Calculate start and end positions
                    start_h = min(i * stride_h, h - kernel_h)
                    start_w = min(j * stride_w, w - kernel_w)
                    end_h = min(start_h + kernel_h, h)
                    end_w = min(start_w + kernel_w, w)
                    
                    # Extract and pool the region
                    region = x[:, :, start_h:end_h, start_w:end_w]
                    pooled = F.adaptive_avg_pool2d(region, 1)
                    pooled_features.append(pooled)
        
        # Concatenate all pooled features
        pooled = torch.cat(pooled_features, dim=1)
        
        # Apply projection
        output = self.projection(pooled)
        
        return output

class SceneClassifier(BaseModel):
    """Scene classifier for understanding image context."""
    
    def __init__(self, config=None):
        """
        Initialize scene classifier.
        
        Args:
            config (dict, optional): Configuration for the classifier
                - backbone: Backbone architecture ('resnet50', 'resnet101')
                - pretrained: Whether to use pretrained weights
                - num_classes: Number of scene categories
        """
        super().__init__(config)
        
        # Default configuration
        self.backbone_type = self.config.get('backbone', 'resnet50')
        self.pretrained = self.config.get('pretrained', True)
        self.num_classes = self.config.get('num_classes', 50)
        
        # Build model
        self.model = self._build_model()
        
        # Scene category names (simplified for demonstration)
        self.categories = self._get_category_names()
    
    def _build_model(self):
        """
        Build scene classification model.
        
        Returns:
            nn.Module: Scene classification model
        """
        # Load pretrained backbone
        if self.backbone_type == 'resnet50':
            backbone = models.resnet50(pretrained=self.pretrained)
            backbone_features = 2048
        elif self.backbone_type == 'resnet101':
            backbone = models.resnet101(pretrained=self.pretrained)
            backbone_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_type}")
        
        # Remove the final FC layer
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Create global context pooling
        context_pooling = GlobalContextPooling(backbone_features, 512)
        
        # Final classifier
        classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
        
        # Combine into a sequential model
        model = nn.Sequential()
        model.add_module('backbone', backbone)
        model.add_module('context_pooling', context_pooling)
        model.add_module('flatten', nn.Flatten())
        model.add_module('classifier', classifier)
        
        return model
    
    def _get_category_names(self):
        """
        Get scene category names.
        
        Returns:
            dict: Mapping from indices to category names
        """
        # This is a simplified list of scene categories
        # In a real implementation, you would load this from a file
        categories = {
            0: 'bedroom',
            1: 'kitchen',
            2: 'living_room',
            3: 'bathroom',
            4: 'office',
            5: 'outdoor_street',
            6: 'outdoor_field',
            7: 'outdoor_forest',
            8: 'outdoor_mountain',
            9: 'outdoor_beach',
            10: 'outdoor_urban',
            11: 'restaurant',
            12: 'store',
            13: 'industrial',
            14: 'sport_field',
            15: 'playground',
            16: 'garden',
            17: 'corridor',
            18: 'conference_room',
            19: 'classroom',
            20: 'library',
            21: 'dining_room',
            22: 'studio',
            23: 'garage',
            24: 'balcony',
            25: 'basement',
            26: 'church',
            27: 'mosque',
            28: 'temple',
            29: 'monastery',
            30: 'station',
            31: 'airport',
            32: 'bus_interior',
            33: 'train_interior',
            34: 'car_interior',
            35: 'pool',
            36: 'waterfall',
            37: 'bridge',
            38: 'plaza',
            39: 'harbor',
            40: 'bar',
            41: 'concert_hall',
            42: 'hospital',
            43: 'warehouse',
            44: 'farm',
            45: 'casino',
            46: 'skyscraper',
            47: 'highway',
            48: 'desert',
            49: 'snow_scene'
        }
        
        return categories
    
    def forward(self, x, return_features=False):
        """
        Forward pass of scene classifier.
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            return_features (bool): Whether to return intermediate features
            
        Returns:
            dict: Classification results with keys:
                - logits: Raw logits
                - probs: Probability distribution
                - scene_idx: Predicted scene index
                - scene_name: Predicted scene name
                - features: Intermediate features (optional)
        """
        # Get backbone features
        if return_features:
            features = self.model.backbone(x)
            pooled = self.model.context_pooling(features)
            flattened = self.model.flatten(pooled)
            logits = self.model.classifier(flattened)
        else:
            # Run full model
            logits = self.model(x)
            features = None
            pooled = None
        
        # Get probabilities and predictions
        probs = F.softmax(logits, dim=1)
        scene_idx = torch.argmax(probs, dim=1)
        
        # Get scene names
        scene_names = [self.categories[idx.item()] for idx in scene_idx]
        
        result = {
            'logits': logits,
            'probs': probs,
            'scene_idx': scene_idx,
            'scene_name': scene_names
        }
        
        if return_features:
            result['backbone_features'] = features
            result['pooled_features'] = pooled
        
        return result
    
    def get_scene_embedding(self, x):
        """
        Extract scene embedding for an input image.
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Scene embedding vectors
        """
        # Extract features before the classifier
        features = self.model.backbone(x)
        pooled = self.model.context_pooling(features)
        embedding = self.model.flatten(pooled)
        
        return embedding
    
    def get_scene_masks(self, scene_idx, height, width):
        """
        Generate scene-specific spatial attention masks.
        
        Args:
            scene_idx (int or torch.Tensor): Scene index
            height (int): Mask height
            width (int): Mask width
            
        Returns:
            torch.Tensor: Attention masks [B, 1, H, W]
        """
        # Convert to list if it's a tensor
        if isinstance(scene_idx, torch.Tensor):
            scene_idx = scene_idx.cpu().tolist()
            if not isinstance(scene_idx, list):
                scene_idx = [scene_idx]
        elif not isinstance(scene_idx, list):
            scene_idx = [scene_idx]
        
        batch_size = len(scene_idx)
        device = next(self.parameters()).device
        
        # Create masks based on scene type
        masks = torch.ones(batch_size, 1, height, width, device=device)
        
        for i, idx in enumerate(scene_idx):
            # Get category name
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            
            category = self.categories.get(idx, 'unknown')
            
            # Apply scene-specific masks
            # This is a simplified version - in a real implementation,
            # you would have more sophisticated scene-specific priors
            
            # For outdoor scenes, emphasize the horizon
            if 'outdoor' in category:
                horizon_y = height // 2
                horizon_band = torch.ones(1, height, width, device=device) * 0.8
                horizon_band[:, horizon_y-height//8:horizon_y+height//8, :] = 1.2
                masks[i] = masks[i] * horizon_band
            
            # For rooms, emphasize the center
            elif any(room in category for room in ['bedroom', 'kitchen', 'living_room', 'bathroom', 'office']):
                y, x = torch.meshgrid(
                    torch.arange(height, device=device),
                    torch.arange(width, device=device)
                )
                center_y, center_x = height // 2, width // 2
                dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
                max_dist = torch.sqrt(torch.tensor(center_y**2 + center_x**2, device=device))
                center_weight = 1 - 0.3 * (dist / max_dist)
                masks[i] = masks[i] * center_weight.unsqueeze(0)
        
        # Normalize masks
        masks = masks / masks.mean(dim=(2, 3), keepdim=True)
        
        return masks 