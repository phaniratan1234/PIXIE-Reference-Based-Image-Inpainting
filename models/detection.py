#!/usr/bin/env python3
"""
Object detection module for image inpainting using YOLOv5.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional
import torchvision.transforms as T
import numpy as np
import cv2
from .base import BaseModel

class YOLODetector(BaseModel):
    """YOLOv5 object detector for semantic understanding."""
    
    def __init__(self, config=None):
        """
        Initialize YOLOv5 detector.
        
        Args:
            config (dict, optional): Configuration for the detector
                - version: YOLOv5 version ('n', 's', 'm', 'l', 'x')
                - pretrained: Whether to use pretrained weights
                - confidence_threshold: Confidence threshold for detections
        """
        super().__init__(config)
        
        # Default configuration
        self.model_version = self.config.get('version', 's')
        self.pretrained = self.config.get('pretrained', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load YOLOv5 model.
        
        Returns:
            torch.nn.Module: YOLOv5 model
        """
        try:
            # Try to load from torch hub
            model = torch.hub.load('ultralytics/yolov5', f'yolov5{self.model_version}', 
                                  pretrained=self.pretrained, verbose=False)
            print(f"Loaded YOLOv5{self.model_version} from torch hub")
        except Exception as e:
            # Fall back to local installation if needed
            print(f"Failed to load YOLOv5 from torch hub: {e}")
            print("Attempting to load YOLOv5 locally...")
            
            try:
                import sys
                from pathlib import Path
                
                # Try to find YOLOv5 installation
                yolo_path = None
                for path in sys.path:
                    if os.path.exists(os.path.join(path, 'yolov5')):
                        yolo_path = os.path.join(path, 'yolov5')
                        break
                
                if yolo_path is None:
                    raise ImportError("YOLOv5 not found in Python path")
                
                sys.path.append(yolo_path)
                from models.common import AutoShape
                from models.yolo import Model
                
                # Load YOLOv5 configuration and weights
                model_path = Path(yolo_path) / 'weights' / f'yolov5{self.model_version}.pt'
                model = torch.load(model_path, map_location=self.device)['model'].float()
                model = AutoShape(model)
                
                print(f"Loaded YOLOv5{self.model_version} locally from {model_path}")
            except Exception as e2:
                # As a last resort, try to automatically install YOLOv5
                print(f"Failed to load YOLOv5 locally: {e2}")
                print("Attempting to install YOLOv5...")
                
                # Use pip to install in a subprocess
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "yolov5"])
                
                # Now try loading again
                model = torch.hub.load('ultralytics/yolov5', f'yolov5{self.model_version}', 
                                      pretrained=self.pretrained, verbose=False)
                print(f"Successfully installed and loaded YOLOv5{self.model_version}")
        
        return model
    
    def forward(self, x, **kwargs):
        """
        Forward pass of YOLOv5 detector.
        
        Args:
            x (torch.Tensor or numpy.ndarray or PIL.Image): Input image
            **kwargs: Additional arguments
                - size (int): Inference size (pixels)
                - augment (bool): Augmented inference
                - visualize (bool): Visualize features
                
        Returns:
            dict: Detection results with keys:
                - boxes: Bounding boxes in format [x1, y1, x2, y2]
                - scores: Confidence scores
                - classes: Class indices
                - labels: Class labels (names)
        """
        # Get additional parameters
        size = kwargs.get('size', 640)
        augment = kwargs.get('augment', False)
        visualize = kwargs.get('visualize', False)
        
        # Set model parameters
        self.model.conf = self.confidence_threshold
        
        # Run detection
        results = self.model(x, size=size, augment=augment, visualize=visualize)
        
        # Process results
        pred = results.pred[0]  # First image in batch
        
        # Extract detections
        boxes = pred[:, :4].cpu().numpy()  # x1, y1, x2, y2
        scores = pred[:, 4].cpu().numpy()
        class_indices = pred[:, 5].cpu().numpy().astype(int)
        
        # Get class labels
        labels = [results.names[i] for i in class_indices]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': class_indices,
            'labels': labels
        }
    
    def get_semantic_map(self, image, output_size=None):
        """
        Generate a semantic segmentation map based on object detections.
        
        Args:
            image (torch.Tensor or numpy.ndarray): Input image
            output_size (tuple, optional): Output size (H, W)
            
        Returns:
            numpy.ndarray: Semantic segmentation map
        """
        # Process image dimensions
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:  # batch
                image = image[0]  # take first image
            
            # Convert to numpy if it's a tensor
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # Denormalize if needed
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image.copy()
        
        # Get original image dimensions
        orig_h, orig_w = image_np.shape[:2]
        
        # Set output size if not provided
        if output_size is None:
            output_size = (orig_h, orig_w)
        
        # Run detection
        detections = self.forward(image_np)
        
        # Create empty semantic map
        semantic_map = np.zeros(output_size, dtype=np.int32)
        
        # Get COCO class indices to match with COCO-Stuff dataset
        coco_class_indices = self.get_coco_indices()
        
        # Draw each detection as a filled polygon
        for i, (box, class_idx, label) in enumerate(zip(detections['boxes'], 
                                                        detections['classes'], 
                                                        detections['labels'])):
            # Convert box to polygon points
            x1, y1, x2, y2 = box
            
            # Scale to output size if needed
            if output_size != (orig_h, orig_w):
                h_scale = output_size[0] / orig_h
                w_scale = output_size[1] / orig_w
                x1, x2 = x1 * w_scale, x2 * w_scale
                y1, y2 = y1 * h_scale, y2 * h_scale
            
            # Convert to integer coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get COCO class index
            coco_idx = coco_class_indices.get(label, i + 1)  # Default to sequential index if not found
            
            # Fill box in semantic map
            semantic_map[y1:y2, x1:x2] = coco_idx
        
        return semantic_map
    
    def get_coco_indices(self):
        """
        Get mapping from YOLO class names to COCO-Stuff class indices.
        
        Returns:
            dict: Mapping from class names to COCO indices
        """
        # This is a simplified mapping for common objects
        # In a real implementation, you would load this from a file or external source
        coco_mapping = {
            'person': 1,
            'bicycle': 2,
            'car': 3,
            'motorcycle': 4,
            'airplane': 5,
            'bus': 6,
            'train': 7,
            'truck': 8,
            'boat': 9,
            'traffic light': 10,
            'fire hydrant': 11,
            'stop sign': 12,
            'parking meter': 13,
            'bench': 14,
            'bird': 15,
            'cat': 16,
            'dog': 17,
            'horse': 18,
            'sheep': 19,
            'cow': 20,
            'elephant': 21,
            'bear': 22,
            'zebra': 23,
            'giraffe': 24,
            'backpack': 25,
            'umbrella': 26,
            'handbag': 27,
            'tie': 28,
            'suitcase': 29,
            'frisbee': 30,
            'skis': 31,
            'snowboard': 32,
            'sports ball': 33,
            'kite': 34,
            'baseball bat': 35,
            'baseball glove': 36,
            'skateboard': 37,
            'surfboard': 38,
            'tennis racket': 39,
            'bottle': 40,
            'wine glass': 41,
            'cup': 42,
            'fork': 43,
            'knife': 44,
            'spoon': 45,
            'bowl': 46,
            'banana': 47,
            'apple': 48,
            'sandwich': 49,
            'orange': 50,
            'broccoli': 51,
            'carrot': 52,
            'hot dog': 53,
            'pizza': 54,
            'donut': 55,
            'cake': 56,
            'chair': 57,
            'couch': 58,
            'potted plant': 59,
            'bed': 60,
            'dining table': 61,
            'toilet': 62,
            'tv': 63,
            'laptop': 64,
            'mouse': 65,
            'remote': 66,
            'keyboard': 67,
            'cell phone': 68,
            'microwave': 69,
            'oven': 70,
            'toaster': 71,
            'sink': 72,
            'refrigerator': 73,
            'book': 74,
            'clock': 75,
            'vase': 76,
            'scissors': 77,
            'teddy bear': 78,
            'hair drier': 79,
            'toothbrush': 80,
        }
        
        return coco_mapping 