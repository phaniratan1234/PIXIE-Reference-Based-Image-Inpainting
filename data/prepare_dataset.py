#!/usr/bin/env python3
"""
Script to prepare the COCO-Stuff dataset for inpainting by:
1. Creating train/val/test splits
2. Resizing images and annotations to a standard size
3. Generating segmentation masks from JSON annotations
4. Saving metadata for efficient loading
"""

import os
import sys
import argparse
import json
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import matplotlib.pyplot as plt
try:
    from pycocotools import mask as coco_mask
except ImportError:
    print("Warning: pycocotools not installed. Install with: pip install pycocotools")
    print("Continuing without pycocotools - RLE mask decoding will not be available")
    
    # Define a dummy coco_mask module with a decode function that raises an exception
    class DummyCocoMask:
        @staticmethod
        def decode(rle):
            raise ImportError("pycocotools is required for RLE mask decoding")
    
    coco_mask = DummyCocoMask()

from skimage import draw
from collections import defaultdict
import random
import gc

def cleanup_data_directory(dir_path):
    """Remove unnecessary files and ensure correct directory structure.
    
    Args:
        dir_path: Base directory to clean
        
    Returns:
        None
    """
    print(f"Cleaning up data directory: {dir_path}")
    
    # Check if annotations folder has images (this should not happen)
    annotations_dir = os.path.join(dir_path, 'annotations')
    if os.path.exists(annotations_dir):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        # Look for image files in annotations directory
        for root, _, files in os.walk(annotations_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if image_files:
            print(f"Warning: Found {len(image_files)} image files in annotations directory.")
            print("These will be moved to the correct images directory.")
            
            # Create images directory if it doesn't exist
            images_dir = os.path.join(dir_path, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # Move image files to images directory
            for img_path in image_files:
                target_path = os.path.join(images_dir, os.path.basename(img_path))
                print(f"Moving {img_path} to {target_path}")
                shutil.move(img_path, target_path)

def process_image(img_id, img_info, img_anns, categories, img_dir, output_dir, split, target_size):
    """
    Process a single image and its annotations.
    
    Args:
        img_id: COCO image ID
        img_info: Dictionary with image information
        img_anns: List of annotations for this image
        categories: Dictionary of category information (mapping of id -> category)
        img_dir: Directory containing source images
        output_dir: Base directory for output
        split: 'train' or 'val'
        target_size: Tuple of (width, height) for resizing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Construct image path
        img_filename = img_info['file_name']
        img_path = os.path.join(img_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping...")
            return False
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        original_size = img.size  # (width, height)
        
        # Initialize mask as zeros (background class = 0)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        if not img_anns:
            print(f"Warning: No annotations for image {img_id}. Skipping...")
            return False
        
        valid_annotations = False
        
        # Process each annotation
        for ann in img_anns:
            if 'segmentation' not in ann:
                continue
                
            # Get category ID and validate
            cat_id = ann['category_id']
            
            # Verify the category ID exists in our categories dictionary
            if cat_id not in categories:
                continue
                
            segmentation = ann['segmentation']
            
            try:
                # Handle different segmentation formats
                if isinstance(segmentation, dict) and 'counts' in segmentation:
                    # RLE format
                    try:
                        binary_mask = coco_mask.decode(segmentation)
                        mask = np.maximum(mask, binary_mask * cat_id)
                        valid_annotations = True
                    except Exception as e:
                        print(f"Error processing RLE annotation for image {img_id}: {e}")
                        continue
                        
                elif isinstance(segmentation, list):
                    # Polygon format
                    if len(segmentation) == 0:
                        continue
                        
                    # Check if it's the right format (list of lists of coordinates)
                    if isinstance(segmentation[0], list) and len(segmentation[0]) >= 6:
                        for polygon in segmentation:
                            if len(polygon) < 6:  # At least 3 points (x,y)
                                continue
                                
                            # Convert polygon to mask
                            try:
                                # Reshape to get x,y coordinates
                                polygon = np.array(polygon).reshape(-1, 2)
                                rr, cc = draw.polygon(polygon[:, 1], polygon[:, 0], (img_info['height'], img_info['width']))
                                mask[rr, cc] = cat_id
                                valid_annotations = True
                            except Exception as e:
                                print(f"Error creating polygon mask for image {img_id}: {e}")
                                continue
                    else:
                        print(f"Warning: Skipping invalid segmentation format for image {img_id}")
                        continue
                else:
                    print(f"Warning: Skipping unknown segmentation format for image {img_id}")
                    continue
            except Exception as e:
                print(f"Error processing annotation for image {img_id}: {e}")
                continue
        
        # Skip if no valid annotations were processed
        if not valid_annotations:
            print(f"No valid annotations for image {img_id}. Skipping...")
            return False
        
        # Convert mask to PIL image
        mask_img = Image.fromarray(mask.astype(np.uint8))
        
        # Resize image and mask to target size
        img_resized = img.resize(target_size, Image.LANCZOS)
        mask_resized = mask_img.resize(target_size, Image.NEAREST)  # Use NEAREST for masks to preserve class IDs
        
        # Create output directories if they don't exist
        img_output_dir = os.path.join(output_dir, split, 'images')
        mask_output_dir = os.path.join(output_dir, split, 'masks')
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(mask_output_dir, exist_ok=True)
        
        # Save processed image and mask
        img_output_path = os.path.join(img_output_dir, f"{img_id}.jpg")
        mask_output_path = os.path.join(mask_output_dir, f"{img_id}.png")
        
        img_resized.save(img_output_path, quality=90)
        mask_resized.save(mask_output_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        return False

def check_coco_json_path(json_path):
    """Find the COCO JSON file from various possible locations."""
    if os.path.exists(json_path):
        return json_path
    
    # Try different possible locations
    possible_paths = [
        json_path,
        os.path.join('data', json_path),
        os.path.join('data', 'coco', json_path),
        os.path.join('data', 'cocostuff-10k-v1.1', json_path)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return json_path  # Return original path, will fail later with appropriate error

def main(args):
    """Process the dataset."""
    # Clean up data directory first
    cleanup_data_directory(os.path.dirname(args.output_dir))
    
    # Find the correct COCO JSON file path
    coco_json_path = check_coco_json_path(args.coco_json)
    
    # Load annotations file
    print(f"Loading COCO annotations from {coco_json_path}")
    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error loading COCO annotations: {e}")
        print(f"Please make sure the file exists at: {coco_json_path}")
        sys.exit(1)
    
    # Get image and annotation info
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    print(f"Loaded {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")
    
    # Create a dictionary of category IDs for quick lookup
    category_ids = {cat['id']: cat for cat in categories}
    print(f"Found {len(category_ids)} unique category IDs")
    
    # Save category info
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'categories.json'), 'w') as f:
        json.dump(categories, f, indent=2)
    
    # Process train and validation splits
    print(f"Processing dataset with target size: {args.img_size}x{args.img_size}")
    
    # Create a map from image id to image info
    image_id_to_info = {img['id']: img for img in images}
    
    # Group annotations by image_id
    image_annotations = defaultdict(list)
    for ann in annotations:
        image_annotations[ann['image_id']].append(ann)
    
    # Get image IDs from the annotations
    image_ids = list(image_annotations.keys())
    
    # Split the dataset
    random.seed(args.seed)  # Use the provided seed
    random.shuffle(image_ids)
    
    val_size = int(len(image_ids) * args.val_split)
    val_ids = set(image_ids[:val_size])
    train_ids = set(image_ids[val_size:])
    
    # Create split directories
    splits = ['train', 'val']
    for split in splits:
        # Only create the correct subdirectories, no annotations folder
        os.makedirs(os.path.join(args.output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'masks'), exist_ok=True)
    
    # Process each split in smaller batches to manage memory better
    batch_size = 200  # Process 200 images at a time
    target_size = (args.img_size, args.img_size)
    
    for split in splits:
        split_ids = list(val_ids if split == 'val' else train_ids)
        print(f"Processing {split} split: {len(split_ids)} images")
        
        # Process in batches
        processed = 0
        failed = 0
        
        for i in range(0, len(split_ids), batch_size):
            batch_ids = split_ids[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(split_ids) + batch_size - 1)//batch_size}, images {i+1}-{min(i+batch_size, len(split_ids))}")
            
            tasks = []
            for img_id in batch_ids:
                if img_id not in image_id_to_info:
                    print(f"Warning: Image ID {img_id} not found in image info")
                    failed += 1
                    continue
                    
                image_info = image_id_to_info[img_id]
                img_annotations = image_annotations[img_id]
                output_dir = os.path.join(args.output_dir, split)
                
                tasks.append((img_id, image_info, img_annotations, category_ids, args.img_dir, output_dir, split, target_size))
            
            # Process batch in parallel
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(process_image, *task) for task in tasks]
                
                batch_success = 0
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing batch"):
                    try:
                        if future.result():
                            batch_success += 1
                            processed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"Error in worker: {e}")
                        failed += 1
            
            # Report progress after each batch
            print(f"Batch complete: {batch_success}/{len(tasks)} images processed successfully")
            
            # Force garbage collection to help with memory management
            gc.collect()
        
        print(f"Split {split} complete: {processed}/{len(split_ids)} images processed successfully, {failed} failed")
    
    # Verify correct directory structure
    for split in splits:
        split_dir = os.path.join(args.output_dir, split)
        annotations_dir = os.path.join(split_dir, 'annotations')
        
        # Check if annotations directory was mistakenly created
        if os.path.exists(annotations_dir):
            print(f"Warning: Found annotations directory in {split_dir}. This should not exist.")
            print("Removing directory...")
            shutil.rmtree(annotations_dir)
    
    print("Dataset preparation complete!")
    print(f"Output saved to {args.output_dir}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare COCO-Stuff dataset for inpainting")
    parser.add_argument('--data-root', type=str, default='./data', 
                        help='Path to the downloaded dataset')
    parser.add_argument('--output-dir', type=str, default='./data/coco-processed', 
                        help='Path to store the processed dataset')
    parser.add_argument('--img-size', type=int, default=512, 
                        help='Size to resize images to')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=4, 
                        help='Number of parallel workers')
    parser.add_argument('--coco-json', type=str, default='cocostuff-10k-v1.1.json', 
                        help='Path to the COCO annotations file')
    parser.add_argument('--val-split', type=float, default=0.1, 
                        help='Ratio of images to use for validation')
    parser.add_argument('--img-dir', type=str, default=None,
                        help='Path to the directory containing the images')
    args = parser.parse_args()
    
    main(args) 