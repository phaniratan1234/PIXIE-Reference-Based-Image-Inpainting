#!/usr/bin/env python3
"""
Preprocessing utilities for image inpainting.
Includes functions for image normalization, augmentation, and conversion.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# Standard normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Custom normalization for our model (using range [-1, 1])
CUSTOM_MEAN = [0.5, 0.5, 0.5]
CUSTOM_STD = [0.5, 0.5, 0.5]

def get_transforms(image_size=512, split='train', normalize=True):
    """
    Get standard image transforms for training or testing.
    
    Args:
        image_size (int): Size to resize images to
        split (str): 'train' or 'val'/'test'
        normalize (bool): Whether to apply normalization
        
    Returns:
        torchvision.transforms.Compose: Transformation pipeline
    """
    transforms_list = []
    
    # Resize
    transforms_list.append(T.Resize((image_size, image_size), interpolation=InterpolationMode.LANCZOS))
    
    # Data augmentation for training
    if split == 'train':
        transforms_list.extend([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        ])
    
    # Convert to tensor
    transforms_list.append(T.ToTensor())
    
    # Normalize
    if normalize:
        transforms_list.append(T.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD))
    
    return T.Compose(transforms_list)

def denormalize(tensor, mean=CUSTOM_MEAN, std=CUSTOM_STD):
    """
    Denormalize a normalized tensor back to [0, 1] range.
    
    Args:
        tensor (torch.Tensor): Normalized image tensor
        mean (list): Mean used for normalization
        std (list): Standard deviation used for normalization
        
    Returns:
        torch.Tensor: Denormalized image tensor in range [0, 1]
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input should be a torch.Tensor, got {type(tensor)}")
    
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    
    return tensor * std + mean

def to_numpy(tensor):
    """
    Convert a torch tensor to numpy array.
    
    Args:
        tensor (torch.Tensor): Input tensor
        
    Returns:
        numpy.ndarray: Numpy array in [0, 255] range and HWC format
    """
    if tensor.ndim == 4:  # batch of images
        tensor = tensor[0]  # take first image in batch
    
    # Denormalize if needed (assuming normalized input)
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = denormalize(tensor)
    
    # Ensure tensor is in CPU and convert to numpy
    array = tensor.detach().cpu().numpy()
    
    # Convert from CHW to HWC and scale to [0, 255]
    array = array.transpose(1, 2, 0) * 255
    return array.astype(np.uint8)

def to_tensor(img, normalize=True):
    """
    Convert a numpy image or PIL image to a PyTorch tensor.
    
    Args:
        img: Numpy array or PIL image
        normalize (bool): Whether to normalize the tensor
        
    Returns:
        torch.Tensor: Image tensor
    """
    if isinstance(img, np.ndarray):
        # Handle numpy array (HWC format)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    elif isinstance(img, Image.Image):
        # Handle PIL image
        img = TF.to_tensor(img)
    else:
        raise TypeError(f"Unsupported input type: {type(img)}")
    
    if normalize:
        img = TF.normalize(img, mean=CUSTOM_MEAN, std=CUSTOM_STD)
    
    return img

def resize_image(img, size, mode='bilinear'):
    """
    Resize an image tensor.
    
    Args:
        img (torch.Tensor): Image tensor [C, H, W] or [B, C, H, W]
        size (tuple): Target size (H, W)
        mode (str): Interpolation mode ('bilinear', 'nearest', 'bicubic')
        
    Returns:
        torch.Tensor: Resized image tensor
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input should be a torch.Tensor, got {type(img)}")
    
    if img.ndim == 3:
        img = img.unsqueeze(0)  # add batch dimension
        was_3d = True
    else:
        was_3d = False
    
    interp_mode = {
        'bilinear': F.interpolate(img, size=size, mode='bilinear', align_corners=False),
        'nearest': F.interpolate(img, size=size, mode='nearest'),
        'bicubic': F.interpolate(img, size=size, mode='bicubic', align_corners=False)
    }
    
    img = interp_mode[mode]
    
    if was_3d:
        img = img.squeeze(0)  # remove batch dimension if input was 3D
    
    return img

def create_edge_map(img, low_threshold=100, high_threshold=200):
    """
    Create an edge map using Canny edge detection.
    
    Args:
        img (numpy.ndarray): Input image in HWC format
        low_threshold (int): Lower threshold for Canny
        high_threshold (int): Higher threshold for Canny
        
    Returns:
        numpy.ndarray: Edge map
    """
    if isinstance(img, torch.Tensor):
        img = to_numpy(img)
    
    # Convert to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    return edges

def create_multi_scale_edge_map(img, scales=(0.5, 1.0, 2.0)):
    """
    Create edge maps at multiple scales and combine them.
    
    Args:
        img (numpy.ndarray): Input image
        scales (tuple): Scales for edge detection
        
    Returns:
        numpy.ndarray: Combined edge map
    """
    if isinstance(img, torch.Tensor):
        img = to_numpy(img)
    
    # Convert to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Create edge maps at different scales
    edge_maps = []
    for scale in scales:
        # Adjust thresholds based on scale
        low_threshold = int(100 * scale)
        high_threshold = int(200 * scale)
        
        # Apply Gaussian blur with scale-dependent sigma
        sigma = 1.0 / scale
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        edge_maps.append(edges)
    
    # Combine edge maps using logical OR
    combined_edges = np.zeros_like(edge_maps[0])
    for edge_map in edge_maps:
        combined_edges = np.logical_or(combined_edges, edge_map)
    
    return combined_edges.astype(np.uint8) * 255

def detect_lines(edge_map, min_line_length=30, max_line_gap=10):
    """
    Detect lines in an edge map using Probabilistic Hough Transform.
    
    Args:
        edge_map (numpy.ndarray): Edge map
        min_line_length (int): Minimum line length
        max_line_gap (int): Maximum gap between line segments
        
    Returns:
        list: Detected lines
    """
    lines = cv2.HoughLinesP(
        edge_map, 
        rho=1,
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=min_line_length, 
        maxLineGap=max_line_gap
    )
    
    return lines if lines is not None else []

def create_line_map(edge_map, shape, min_line_length=30, max_line_gap=10):
    """
    Create a line map from an edge map.
    
    Args:
        edge_map (numpy.ndarray): Edge map
        shape (tuple): Output shape (H, W)
        min_line_length (int): Minimum line length
        max_line_gap (int): Maximum gap between line segments
        
    Returns:
        numpy.ndarray: Line map
    """
    lines = detect_lines(edge_map, min_line_length, max_line_gap)
    
    line_map = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_map, (x1, y1), (x2, y2), 255, 1)
    
    return line_map

def extract_patches(img, patch_size=256, stride=128):
    """
    Extract overlapping patches from an image.
    
    Args:
        img (torch.Tensor): Input image tensor [C, H, W]
        patch_size (int): Size of patches
        stride (int): Stride between patches
        
    Returns:
        torch.Tensor: Tensor of patches [N, C, patch_size, patch_size]
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input should be a torch.Tensor, got {type(img)}")
    
    if img.ndim == 4:  # batch of images
        img = img[0]  # take first image in batch
    
    C, H, W = img.shape
    
    # Calculate the number of patches in each dimension
    num_h = (H - patch_size) // stride + 1
    num_w = (W - patch_size) // stride + 1
    
    patches = []
    for i in range(num_h):
        for j in range(num_w):
            h_start = i * stride
            w_start = j * stride
            patch = img[:, h_start:h_start+patch_size, w_start:w_start+patch_size]
            patches.append(patch)
    
    return torch.stack(patches)

def reconstruct_from_patches(patches, original_size, stride=128):
    """
    Reconstruct an image from overlapping patches.
    
    Args:
        patches (torch.Tensor): Tensor of patches [N, C, patch_size, patch_size]
        original_size (tuple): Original image size (H, W)
        stride (int): Stride used for extraction
        
    Returns:
        torch.Tensor: Reconstructed image [C, H, W]
    """
    if not isinstance(patches, torch.Tensor):
        raise TypeError(f"Input should be a torch.Tensor, got {type(patches)}")
    
    N, C, patch_size, _ = patches.shape
    H, W = original_size
    
    # Calculate the number of patches in each dimension
    num_h = (H - patch_size) // stride + 1
    num_w = (W - patch_size) // stride + 1
    
    # Create empty tensors for reconstructed image and weight map
    reconstructed = torch.zeros(C, H, W, device=patches.device)
    weight_map = torch.zeros(1, H, W, device=patches.device)
    
    # Create a weight tensor for blending overlapping patches
    patch_weight = torch.ones(1, patch_size, patch_size, device=patches.device)
    
    patch_idx = 0
    for i in range(num_h):
        for j in range(num_w):
            h_start = i * stride
            w_start = j * stride
            
            # Extract patch
            patch = patches[patch_idx]
            
            # Add patch to reconstructed image with weights
            reconstructed[:, h_start:h_start+patch_size, w_start:w_start+patch_size] += patch
            weight_map[:, h_start:h_start+patch_size, w_start:w_start+patch_size] += patch_weight
            
            patch_idx += 1
    
    # Normalize by weights
    weight_map = torch.clamp(weight_map, min=1e-8)
    reconstructed = reconstructed / weight_map
    
    return reconstructed 