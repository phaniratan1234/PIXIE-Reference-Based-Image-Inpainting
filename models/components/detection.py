"""
Object Detection Module

This module contains the ObjectDetector component which identifies and localizes
objects in images for inpainting purposes. It detects objects both in the known
regions and predicts potential objects in masked areas to guide the inpainting
process with appropriate object placement and boundaries.

The detector uses a lightweight architecture to generate:
1. Bounding boxes for objects
2. Object class predictions
3. Object boundary maps that indicate object edges
4. Semantic region proposals for improved object coherence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math

class ObjectDetector(nn.Module):
    """
    ObjectDetector identifies and localizes objects in images for inpainting guidance.
    
    This component is responsible for:
    
    1. Detecting visible objects in the unmasked image regions
    2. Inferring likely object boundaries and continuations in masked regions
    3. Generating boundary maps to guide structure preservation during inpainting
    4. Providing object context information to enhance semantic coherence
    
    The detector uses a feature pyramid network (FPN) architecture for multi-scale
    object detection, with specialized heads for different output types.
    """
    
    def __init__(
        self,
        in_channels: int,
        feature_channels: int = 64,
        num_classes: int = 80,
        num_fpn_levels: int = 3,
        use_spectral_norm: bool = True,
        **kwargs
    ):
        """
        Initialize the object detector.
        
        Args:
            in_channels: Number of input channels
            feature_channels: Base number of feature channels
            num_classes: Number of object classes to detect
            num_fpn_levels: Number of levels in the feature pyramid network
            use_spectral_norm: Whether to use spectral normalization on convolutions
            **kwargs: Additional arguments including 'device' specification
        """
        super().__init__()
        
        # Store configuration parameters
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.num_classes = num_classes
        self.num_fpn_levels = num_fpn_levels
        self.use_spectral_norm = use_spectral_norm
        
        # Define normalization function based on configuration
        self.norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Backbone network - extracts hierarchical features
        self.backbone = self._build_backbone()
        
        # Feature pyramid network - combines features across scales
        self.fpn = self._build_fpn()
        
        # Detection heads
        self.class_head = self._build_class_head()
        self.box_head = self._build_box_head()
        self.boundary_head = self._build_boundary_head()
        self.mask_head = self._build_mask_head()
        
        # Initialize weights
        self._init_weights()
        
        # Move to specified device if provided
        if 'device' in kwargs:
            self.to(kwargs['device'])
    
    def _build_backbone(self) -> nn.ModuleList:
        """
        Build the backbone feature extraction network.
        
        Returns:
            ModuleList containing sequential layers of the backbone network
        """
        layers = []
        current_channels = self.in_channels
        
        # Create a 4-stage backbone with increasing channel counts
        for i in range(4):
            out_channels = self.feature_channels * (2 ** i)
            layers.append(self._make_backbone_layer(current_channels, out_channels, stride=2))
            current_channels = out_channels
            
        return nn.ModuleList(layers)
    
    def _make_backbone_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """
        Create a backbone layer with specified channels and stride.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution
            
        Returns:
            Sequential module for the backbone layer
        """
        return nn.Sequential(
            self.norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self.norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_fpn(self) -> nn.ModuleDict:
        """
        Build the Feature Pyramid Network for multi-scale feature fusion.
        
        Returns:
            ModuleDict containing lateral and output convolutions for the FPN
        """
        fpn_modules = {}
        
        # Lateral connections convert all backbone features to the same channel count
        for i in range(1, self.num_fpn_levels + 1):
            level = 4 - i  # Start from top level (smallest spatial size)
            in_channels = self.feature_channels * (2 ** level)
            fpn_modules[f'lateral_{level}'] = nn.Conv2d(in_channels, self.feature_channels, kernel_size=1)
            
        # Output convolutions apply 3x3 convs to each merged feature level
        for i in range(self.num_fpn_levels):
            level = 4 - i - 1  # FPN level naming
            fpn_modules[f'output_{level}'] = nn.Sequential(
                self.norm_layer(nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=1)),
                nn.InstanceNorm2d(self.feature_channels),
                nn.ReLU(inplace=True)
            )
            
        return nn.ModuleDict(fpn_modules)
    
    def _build_class_head(self) -> nn.ModuleList:
        """
        Build the class prediction head for object classification.
        
        Returns:
            ModuleList containing class prediction layers for each FPN level
        """
        head_modules = []
        
        # Create classification head for each FPN level
        for _ in range(self.num_fpn_levels):
            head = nn.Sequential(
                self.norm_layer(nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=1)),
                nn.InstanceNorm2d(self.feature_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feature_channels, self.num_classes, kernel_size=1)
            )
            head_modules.append(head)
            
        return nn.ModuleList(head_modules)
    
    def _build_box_head(self) -> nn.ModuleList:
        """
        Build the bounding box regression head.
        
        Returns:
            ModuleList containing box regression layers for each FPN level
        """
        head_modules = []
        
        # Create box regression head for each FPN level
        for _ in range(self.num_fpn_levels):
            head = nn.Sequential(
                self.norm_layer(nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=1)),
                nn.InstanceNorm2d(self.feature_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feature_channels, 4, kernel_size=1)  # 4 values: [x, y, w, h]
            )
            head_modules.append(head)
            
        return nn.ModuleList(head_modules)
    
    def _build_boundary_head(self) -> nn.Sequential:
        """
        Build the boundary detection head for object edge prediction.
        
        Returns:
            Sequential module for object boundary prediction
        """
        return nn.Sequential(
            self.norm_layer(nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(self.feature_channels),
            nn.ReLU(inplace=True),
            self.norm_layer(nn.Conv2d(self.feature_channels, self.feature_channels // 2, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(self.feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _build_mask_head(self) -> nn.Sequential:
        """
        Build the instance segmentation head for fine-grained object masks.
        
        Returns:
            Sequential module for instance segmentation mask prediction
        """
        return nn.Sequential(
            self.norm_layer(nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(self.feature_channels),
            nn.ReLU(inplace=True),
            self.norm_layer(nn.Conv2d(self.feature_channels, self.feature_channels // 2, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(self.feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels // 2, self.num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _init_weights(self):
        """
        Initialize model weights for better training dynamics.
        
        Uses Kaiming initialization for convolutional layers and
        zeros for biases.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _extract_backbone_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract hierarchical features from the backbone network.
        
        Args:
            x: Input tensor of images [B, C, H, W]
            
        Returns:
            List of feature tensors from each level of the backbone
        """
        features = []
        
        for layer in self.backbone:
            x = layer(x)
            features.append(x)
            
        return features
    
    def _build_feature_pyramid(self, backbone_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Construct FPN features from backbone features using top-down pathway.
        
        Args:
            backbone_features: List of features from the backbone network
            
        Returns:
            List of feature tensors from the FPN
        """
        fpn_features = []
        
        # Start with the top level (smallest spatial dimension)
        top_level = 4 - self.num_fpn_levels
        prev_feature = self.fpn[f'lateral_{top_level}'](backbone_features[top_level])
        fpn_features.append(self.fpn[f'output_{top_level}'](prev_feature))
        
        # Process other FPN levels with lateral connections and upsampling
        for i in range(1, self.num_fpn_levels):
            level = top_level - i
            lateral = self.fpn[f'lateral_{level}'](backbone_features[level])
            
            # Upsample the previous feature and add to current lateral connection
            prev_feature = F.interpolate(prev_feature, size=lateral.shape[2:], mode='bilinear', align_corners=False)
            merged_feature = lateral + prev_feature
            
            # Store upsampled feature for next iteration
            prev_feature = merged_feature
            
            # Apply output convolution and store in result list
            fpn_features.append(self.fpn[f'output_{level}'](merged_feature))
            
        # Reverse to have coarse-to-fine order (higher resolution first)
        fpn_features.reverse()
        
        return fpn_features
    
    def forward(
        self, 
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Process input images for object detection.
        
        Args:
            x: Input tensor of images [B, C, H, W]
            masks: Optional binary masks indicating regions to inpaint [B, 1, H, W]
                  1 = area to inpaint (hole), 0 = known region
            return_features: Whether to return intermediate feature maps
            
        Returns:
            Dictionary containing:
            - 'class_logits': List of class prediction tensors for each FPN level
            - 'box_preds': List of box prediction tensors for each FPN level
            - 'boundary_map': Object boundary prediction map
            - 'mask_preds': Instance segmentation predictions
            If return_features is True, additional keys:
            - 'backbone_features': List of feature tensors from the backbone
            - 'fpn_features': List of feature tensors from the FPN
        """
        # Concatenate masks if provided
        if masks is not None:
            x = torch.cat([x, masks], dim=1)
        
        # Extract backbone features
        backbone_features = self._extract_backbone_features(x)
        
        # Build FPN features
        fpn_features = self._build_feature_pyramid(backbone_features)
        
        # Apply detection heads to each FPN level
        class_logits = []
        box_preds = []
        for i, feature in enumerate(fpn_features):
            class_logits.append(self.class_head[i](feature))
            box_preds.append(self.box_head[i](feature))
        
        # Apply boundary and mask heads to the highest resolution FPN feature
        boundary_map = self.boundary_head(fpn_features[0])
        mask_preds = self.mask_head(fpn_features[0])
        
        # Prepare return dictionary
        output = {
            'class_logits': class_logits,
            'box_preds': box_preds,
            'boundary_map': boundary_map,
            'mask_preds': mask_preds
        }
        
        # Add features if requested
        if return_features:
            output['backbone_features'] = backbone_features
            output['fpn_features'] = fpn_features
        
        return output
    
    def detect_objects(
        self, 
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in images and return processed results.
        
        Args:
            images: Input tensor of images [B, 3, H, W]
            masks: Optional binary masks indicating regions to inpaint [B, 1, H, W]
            confidence_threshold: Minimum confidence for detection to be kept
            
        Returns:
            List of dictionaries, one per image, containing:
            - 'boxes': Tensor of bounding boxes [N, 4] (x1, y1, x2, y2 format)
            - 'scores': Tensor of confidence scores [N]
            - 'labels': Tensor of class labels [N]
            - 'boundary_map': Object boundary prediction map [H, W]
            - 'masks': Instance segmentation masks [N, H, W]
        """
        # Run model forward
        with torch.no_grad():
            outputs = self.forward(images, masks)
        
        batch_size = images.shape[0]
        results = []
        
        # Process each image in the batch
        for batch_idx in range(batch_size):
            # Initialize lists for detections
            all_boxes = []
            all_scores = []
            all_labels = []
            
            # Process each FPN level
            for level_idx, (cls_preds, box_preds) in enumerate(zip(outputs['class_logits'], outputs['box_preds'])):
                # Get predictions for current image
                cls_pred = cls_preds[batch_idx]  # [num_classes, H, W]
                box_pred = box_preds[batch_idx]  # [4, H, W]
                
                # Find locations with high confidence
                cls_pred_t = cls_pred.permute(1, 2, 0)  # [H, W, num_classes]
                confidence, class_idx = torch.max(cls_pred_t, dim=2)  # [H, W]
                
                # Filter by confidence threshold
                mask = confidence > confidence_threshold
                y_indices, x_indices = torch.where(mask)
                
                if len(y_indices) > 0:
                    # Get box predictions at selected locations
                    box_values = box_pred[:, y_indices, x_indices].permute(1, 0)  # [N, 4]
                    
                    # Convert from [x, y, w, h] to [x1, y1, x2, y2]
                    boxes = torch.zeros_like(box_values)
                    boxes[:, 0] = box_values[:, 0] - box_values[:, 2] / 2  # x1 = x - w/2
                    boxes[:, 1] = box_values[:, 1] - box_values[:, 3] / 2  # y1 = y - h/2
                    boxes[:, 2] = box_values[:, 0] + box_values[:, 2] / 2  # x2 = x + w/2
                    boxes[:, 3] = box_values[:, 1] + box_values[:, 3] / 2  # y2 = y + h/2
                    
                    # Adjust coordinates based on FPN level
                    scale_factor = 2 ** level_idx
                    boxes *= scale_factor
                    
                    # Add to detection lists
                    all_boxes.append(boxes)
                    all_scores.append(confidence[mask])
                    all_labels.append(class_idx[mask])
            
            # Combine detections across FPN levels
            if all_boxes:
                boxes = torch.cat(all_boxes, dim=0)
                scores = torch.cat(all_scores, dim=0)
                labels = torch.cat(all_labels, dim=0)
                
                # Apply non-maximum suppression (simplified)
                keep = self._nms(boxes, scores, iou_threshold=0.5)
                
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            else:
                # No detections
                boxes = torch.zeros((0, 4), device=images.device)
                scores = torch.zeros(0, device=images.device)
                labels = torch.zeros(0, dtype=torch.long, device=images.device)
            
            # Get boundary map and instance masks for current image
            boundary_map = outputs['boundary_map'][batch_idx, 0]  # [H, W]
            
            # Create instance masks from mask predictions
            masks = []
            if labels.numel() > 0:
                for label_idx in range(len(labels)):
                    class_idx = labels[label_idx].item()
                    mask = outputs['mask_preds'][batch_idx, class_idx]  # [H, W]
                    masks.append(mask)
                
                # Stack masks if any exist
                if masks:
                    masks = torch.stack(masks)
                else:
                    masks = torch.zeros((0, images.shape[2], images.shape[3]), device=images.device)
            else:
                masks = torch.zeros((0, images.shape[2], images.shape[3]), device=images.device)
            
            # Add results for current image
            results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
                'boundary_map': boundary_map,
                'masks': masks
            })
        
        return results
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """
        Perform non-maximum suppression on bounding boxes.
        
        Args:
            boxes: Bounding boxes in [x1, y1, x2, y2] format [N, 4]
            scores: Confidence scores [N]
            iou_threshold: IoU threshold for overlap
            
        Returns:
            Tensor of indices to keep
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        
        # Convert to [x1, y1, x2, y2] if not already
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by score
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
                
            i = order[0].item()
            keep.append(i)
            
            # Calculate IoU with rest of boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h
            
            # IoU = intersection / union
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU less than threshold
            inds = torch.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
            
        return torch.tensor(keep, dtype=torch.long, device=boxes.device) 