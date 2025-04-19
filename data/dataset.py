#!/usr/bin/env python3
"""
Dataset classes for image inpainting training and evaluation.
Includes specialized dataset handling for COCO-Stuff with advanced mask generation.
"""

import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path
import cv2
from .mask_generator import get_mask_generator
from torchvision import transforms
import glob
from typing import Dict, List, Optional, Tuple, Union

class InpaintingDataset(Dataset):
    """Dataset for training image inpainting models with various mask types."""
    
    def __init__(
        self, 
        data_root: str,
        split: str = "train",
        image_size: int = 256,
        mask_type: str = "random",
        augmentation: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_root: Root directory of the dataset
            split: 'train' or 'val'
            image_size: Size to resize images to
            mask_type: Type of mask to use - 'random', 'bbox', 'segmentation'
            augmentation: Whether to use data augmentation
            max_samples: Maximum number of samples to use (for limiting dataset size)
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.mask_type = mask_type
        self.augmentation = augmentation
        
        # Get image paths based on dataset structure
        if os.path.exists(os.path.join(data_root, split, 'images')):
            # COCO-Stuff processed structure (from prepare_dataset.py)
            self.image_paths = self._get_image_paths(os.path.join(data_root, split, 'images'))
            self.annotation_dir = os.path.join(data_root, split, 'annotations')
            self.has_annotations = True
        elif os.path.exists(os.path.join(data_root, split)):
            # Generic structure with just images
            self.image_paths = self._get_image_paths(os.path.join(data_root, split))
            self.has_annotations = False
        else:
            # Fall back to recursive search
            self.image_paths = self._get_image_paths_recursive(data_root)
            self.has_annotations = False
        
        if max_samples is not None and max_samples < len(self.image_paths):
            self.image_paths = self.image_paths[:max_samples]
            
        print(f"Loaded {len(self.image_paths)} images for {split} split")
        print(f"Annotation availability: {self.has_annotations}")
        
        # Load label map if available
        self.label_map = None
        label_map_path = os.path.join(data_root, 'label_map.json')
        if os.path.exists(label_map_path):
            try:
                with open(label_map_path, 'r') as f:
                    self.label_map = json.load(f)
                print(f"Loaded label map with {len(self.label_map)} categories")
            except Exception as e:
                print(f"Failed to load label map: {e}")
        
        # Define transformations
        self._setup_transforms()
    
    def _get_image_paths(self, directory: str) -> List[str]:
        """Get all image paths in a directory."""
        extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, ext)))
        return sorted(image_paths)
    
    def _get_image_paths_recursive(self, directory: str) -> List[str]:
        """Get all image paths recursively in a directory and its subdirectories."""
        extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
        return sorted(image_paths)
        
    def _setup_transforms(self):
        """Setup image transformations."""
        # Basic transforms for both train and val
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        
        # Augmentation for training
        if self.split == 'train' and self.augmentation:
            augmentation_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            ]
            transform_list = augmentation_transforms + transform_list
            
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _get_random_mask(self) -> torch.Tensor:
        """Generate random irregular mask."""
        mask = torch.ones((1, self.image_size, self.image_size), dtype=torch.float32)
        
        # Number of strokes for the mask
        num_strokes = random.randint(5, 20)
        
        for _ in range(num_strokes):
            # Random starting point
            x1, y1 = random.randint(0, self.image_size-1), random.randint(0, self.image_size-1)
            
            # Random stroke params
            stroke_width = random.randint(10, 40)
            stroke_length = random.randint(30, 100)
            angle = random.uniform(0, 2 * np.pi)
            
            # Calculate ending point
            x2 = min(max(0, int(x1 + stroke_length * np.cos(angle))), self.image_size-1)
            y2 = min(max(0, int(y1 + stroke_length * np.sin(angle))), self.image_size-1)
            
            # Draw line
            for i in range(min(stroke_length, self.image_size)):
                x = min(int(x1 + i * (x2 - x1) / stroke_length), self.image_size-1)
                y = min(int(y1 + i * (y2 - y1) / stroke_length), self.image_size-1)
                
                # Draw circle at this point
                for dx in range(-stroke_width, stroke_width + 1):
                    for dy in range(-stroke_width, stroke_width + 1):
                        if dx*dx + dy*dy <= stroke_width*stroke_width:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.image_size and 0 <= ny < self.image_size:
                                mask[0, ny, nx] = 0
        
        return mask
    
    def _get_bbox_mask(self) -> torch.Tensor:
        """Generate rectangular mask."""
        mask = torch.ones((1, self.image_size, self.image_size), dtype=torch.float32)
        
        # Determine box size (25-50% of image size)
        h_ratio = random.uniform(0.25, 0.5)
        w_ratio = random.uniform(0.25, 0.5)
        box_h = int(self.image_size * h_ratio)
        box_w = int(self.image_size * w_ratio)
        
        # Determine top-left corner
        top = random.randint(0, self.image_size - box_h)
        left = random.randint(0, self.image_size - box_w)
        
        # Create mask (0 for masked region, 1 for valid region)
        mask[0, top:top+box_h, left:left+box_w] = 0
        
        return mask
    
    def _get_segmentation_mask(self, img_path: str = None) -> torch.Tensor:
        """
        Generate mask based on segmentation data if available,
        otherwise create synthetic segmentation-like mask.
        """
        mask = torch.ones((1, self.image_size, self.image_size), dtype=torch.float32)
        
        # Try to use real segmentation if available
        if self.has_annotations and img_path:
            try:
                # Extract image filename and get corresponding annotation
                img_filename = os.path.basename(img_path)
                anno_path = os.path.join(self.annotation_dir, img_filename)
                
                if os.path.exists(anno_path):
                    # Load segmentation map
                    seg_map = Image.open(anno_path)
                    seg_map = seg_map.resize((self.image_size, self.image_size), Image.NEAREST)
                    seg_map = np.array(seg_map)
                    
                    # Pick a random label to remove
                    unique_labels = np.unique(seg_map)
                    # Skip background (typically 0)
                    unique_labels = unique_labels[unique_labels > 0]
                    
                    if len(unique_labels) > 0:
                        target_label = np.random.choice(unique_labels)
                        # Create mask where this label appears
                        label_mask = (seg_map == target_label)
                        
                        # Convert to tensor format (0 for masked region, 1 for valid region)
                        mask = torch.ones((1, self.image_size, self.image_size), dtype=torch.float32)
                        mask[0, label_mask] = 0
                        
                        # Ensure mask covers sufficient area
                        if mask.sum() > 0.9 * self.image_size * self.image_size:
                            # Fallback to synthetic if segmentation region is too small
                            return self._create_synthetic_segmentation_mask()
                        
                        return mask
            except Exception as e:
                print(f"Error using segmentation mask for {img_path}: {e}")
                
        # Fallback to synthetic segmentation-like mask
        return self._create_synthetic_segmentation_mask()
    
    def _create_synthetic_segmentation_mask(self) -> torch.Tensor:
        """Create a synthetic mask that resembles object segmentation."""
        mask = torch.ones((1, self.image_size, self.image_size), dtype=torch.float32)
        
        # Number of blobs
        num_blobs = random.randint(1, 3)
        
        for _ in range(num_blobs):
            # Random center
            cx = random.randint(0, self.image_size-1)
            cy = random.randint(0, self.image_size-1)
            
            # Random radius range
            min_rad = int(self.image_size * 0.1)
            max_rad = int(self.image_size * 0.25)
            radius = random.randint(min_rad, max_rad)
            
            # Create irregular blob
            num_points = random.randint(8, 15)
            angles = sorted([random.uniform(0, 2 * np.pi) for _ in range(num_points)])
            
            # Get points on irregular perimeter
            points = []
            for angle in angles:
                r = radius * random.uniform(0.8, 1.2)
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                points.append((x, y))
            
            # Convert points to numpy arrays for easier operations
            points = np.array(points)
            
            # Fill the polygon
            xx, yy = np.meshgrid(range(self.image_size), range(self.image_size))
            points_mesh = np.column_stack((xx.ravel(), yy.ravel()))
            
            from matplotlib.path import Path
            path = Path(points)
            mask_blob = path.contains_points(points_mesh).reshape(self.image_size, self.image_size)
            
            # Apply to mask
            mask[0, mask_blob] = 0
            
        return mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            dict with keys:
                'image': Original image tensor
                'masked_image': Image with masked region
                'mask': Binary mask (0 for masked region, 1 for valid region)
                'path': Path to the original image
        """
        img_path = self.image_paths[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        # Get mask based on specified type
        if self.mask_type == 'random':
            mask = self._get_random_mask()
        elif self.mask_type == 'bbox':
            mask = self._get_bbox_mask()
        elif self.mask_type == 'segmentation':
            mask = self._get_segmentation_mask(img_path)
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")
        
        # Apply mask to image
        masked_img = img_tensor * mask
        
        return {
            'image': img_tensor,
            'masked_image': masked_img,
            'mask': mask,
            'path': img_path
        }

def build_dataloader(
    config: Dict,
    split: str = "train",
    shuffle: bool = None,
    num_workers: int = 4
) -> DataLoader:
    """
    Build a dataloader for the inpainting dataset.
    
    Args:
        config: Configuration dictionary
        split: 'train' or 'val'
        shuffle: Whether to shuffle the data. If None, will shuffle for train split only.
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader for the specified dataset
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset_cfg = config.get('dataset', {}).get(split, {})
    data_root = config.get('data_root', './data/coco-processed')
    
    dataset = InpaintingDataset(
        data_root=data_root,
        split=split,
        image_size=dataset_cfg.get('image_size', 256),
        mask_type=dataset_cfg.get('mask_type', 'random'),
        augmentation=dataset_cfg.get('augmentation', split == 'train'),
        max_samples=dataset_cfg.get('max_samples')
    )
    
    batch_size = dataset_cfg.get('batch_size', config.get('batch_size', 16))
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader 