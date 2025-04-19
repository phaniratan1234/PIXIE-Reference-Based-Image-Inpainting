"""
Scene Classifier Module

This module contains the SceneClassifier component which analyzes and classifies
the scene content of images for inpainting. It categorizes image regions based on
scene types (indoor, outdoor, natural, urban, etc.) and content types to help guide
the inpainting process with appropriate stylistic and contextual information.

The classifier uses a CNN-based architecture to extract features and predict
scene and content classifications that inform the inpainting process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union

class SceneClassifier(nn.Module):
    """
    SceneClassifier analyzes and classifies image scenes to guide inpainting.
    
    This component is responsible for:
    
    1. Classifying scene types (indoor, outdoor, natural, urban, etc.)
    2. Detecting content categories (people, objects, landscape elements)
    3. Extracting scene-specific features for style-aware inpainting
    4. Providing scene context to other components in the inpainting pipeline
    
    The classifier uses a ResNet-based backbone with specialized classification
    heads for different aspects of scene understanding.
    """
    
    def __init__(
        self,
        in_channels: int,
        feature_channels: int = 64,
        num_scene_classes: int = 5,
        num_content_classes: int = 10,
        use_spectral_norm: bool = True,
        **kwargs
    ):
        """
        Initialize the scene classifier.
        
        Args:
            in_channels: Number of input channels
            feature_channels: Base number of feature channels
            num_scene_classes: Number of scene categories to classify
            num_content_classes: Number of content categories to classify
            use_spectral_norm: Whether to use spectral normalization on convolutions
            **kwargs: Additional arguments including 'device' specification
        """
        super().__init__()
        
        # Store configuration parameters
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.num_scene_classes = num_scene_classes
        self.num_content_classes = num_content_classes
        self.use_spectral_norm = use_spectral_norm
        
        # Define normalization function based on configuration
        self.norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            self._make_resblock(in_channels, feature_channels, stride=2),
            self._make_resblock(feature_channels, feature_channels * 2, stride=2),
            self._make_resblock(feature_channels * 2, feature_channels * 4, stride=2),
            self._make_resblock(feature_channels * 4, feature_channels * 8, stride=2)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Scene classification head
        self.scene_classifier = nn.Sequential(
            self.norm_layer(nn.Linear(feature_channels * 8, feature_channels * 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_channels * 4, num_scene_classes)
        )
        
        # Content classification head (multi-label)
        self.content_classifier = nn.Sequential(
            self.norm_layer(nn.Linear(feature_channels * 8, feature_channels * 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_channels * 4, num_content_classes)
        )
        
        # Style embedding generator
        self.style_embedding = nn.Sequential(
            self.norm_layer(nn.Linear(feature_channels * 8, feature_channels * 2)),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels * 2, feature_channels)
        )
        
        # Scene feature extractor (returns spatial features)
        self.scene_features = nn.Sequential(
            self.norm_layer(nn.Conv2d(feature_channels * 8, feature_channels * 4, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(feature_channels * 4),
            nn.ReLU(inplace=True),
            self.norm_layer(nn.Conv2d(feature_channels * 4, feature_channels * 2, kernel_size=1))
        )
        
        # Initialize weights
        self._init_weights()
        
        # Move to specified device if provided
        if 'device' in kwargs:
            self.to(kwargs['device'])
    
    def _make_resblock(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
        """
        Create a residual block for feature extraction.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution
            
        Returns:
            Sequential module for the residual block
        """
        # Shortcut connection
        shortcut = nn.Sequential(
            self.norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
            nn.InstanceNorm2d(out_channels)
        )
        
        # Main branch
        layers = nn.Sequential(
            self.norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self.norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.InstanceNorm2d(out_channels)
        )
        
        # Residual connection wrapper
        class ResBlock(nn.Module):
            def __init__(self, layers, shortcut):
                super().__init__()
                self.layers = layers
                self.shortcut = shortcut
                self.relu = nn.ReLU(inplace=True)
                
            def forward(self, x):
                identity = self.shortcut(x)
                out = self.layers(x)
                out += identity
                out = self.relu(out)
                return out
        
        return ResBlock(layers, shortcut)
    
    def _init_weights(self):
        """
        Initialize model weights for better training dynamics.
        
        Uses Kaiming initialization for convolutional layers, Xavier for linear layers,
        and zeros for biases.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process input images for scene classification.
        
        Args:
            x: Input tensor of images [B, C, H, W]
            masks: Optional binary masks indicating regions to inpaint [B, 1, H, W]
                  1 = area to inpaint (hole), 0 = known region
            return_features: Whether to return intermediate feature maps
            
        Returns:
            Dictionary containing:
            - 'scene_logits': Scene classification logits [B, num_scene_classes]
            - 'scene_probs': Scene classification probabilities [B, num_scene_classes]
            - 'content_logits': Content classification logits [B, num_content_classes]
            - 'content_probs': Content classification probabilities [B, num_content_classes]
            - 'style_embedding': Style embedding vector [B, feature_channels]
            - 'scene_features': Spatial scene features [B, feature_channels*2, H/16, W/16]
            If return_features is True, additional key:
            - 'backbone_features': Feature tensor from backbone [B, feature_channels*8, H/16, W/16]
        """
        # Concatenate masks if provided
        if masks is not None:
            x = torch.cat([x, masks], dim=1)
        
        # Extract features through backbone
        features = self.backbone(x)
        
        # Global pooling for classification
        pooled_features = self.global_pool(features).flatten(1)
        
        # Scene classification
        scene_logits = self.scene_classifier(pooled_features)
        scene_probs = F.softmax(scene_logits, dim=1)
        
        # Content classification (multi-label)
        content_logits = self.content_classifier(pooled_features)
        content_probs = torch.sigmoid(content_logits)
        
        # Generate style embedding
        style_embedding = self.style_embedding(pooled_features)
        
        # Extract spatial scene features
        scene_features = self.scene_features(features)
        
        # Prepare return dictionary
        output = {
            'scene_logits': scene_logits,
            'scene_probs': scene_probs,
            'content_logits': content_logits,
            'content_probs': content_probs,
            'style_embedding': style_embedding,
            'scene_features': scene_features
        }
        
        # Add backbone features if requested
        if return_features:
            output['backbone_features'] = features
        
        return output
    
    def classify_scene(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Wrapper function for simple scene classification of images.
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            Dictionary with scene classification results
        """
        return self.forward(images, return_features=False)
    
    def get_scene_labels(self, probs: torch.Tensor, threshold: float = 0.5) -> List[List[int]]:
        """
        Convert scene probability predictions to class labels.
        
        Args:
            probs: Scene classification probabilities [B, num_scene_classes]
            threshold: Probability threshold for positive classification
            
        Returns:
            List of lists containing class indices with probabilities > threshold
        """
        batch_labels = []
        
        # For each image in the batch
        for image_probs in probs:
            # Get indices where probability exceeds threshold
            indices = torch.where(image_probs > threshold)[0].tolist()
            
            # If no class exceeds threshold, use argmax
            if not indices:
                indices = [torch.argmax(image_probs).item()]
                
            batch_labels.append(indices)
            
        return batch_labels
    
    def get_content_labels(self, probs: torch.Tensor, threshold: float = 0.5) -> List[List[int]]:
        """
        Convert content probability predictions to class labels.
        
        Args:
            probs: Content classification probabilities [B, num_content_classes]
            threshold: Probability threshold for positive classification
            
        Returns:
            List of lists containing class indices with probabilities > threshold
        """
        batch_labels = []
        
        # For each image in the batch
        for image_probs in probs:
            # Get indices where probability exceeds threshold
            indices = torch.where(image_probs > threshold)[0].tolist()
            batch_labels.append(indices)
            
        return batch_labels 