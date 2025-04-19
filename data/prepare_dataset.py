#!/usr/bin/env python3
"""
Script to prepare the COCO-Stuff dataset for inpainting by:
1. Creating train/val/test splits
2. Resizing images and annotations to a standard size
3. Saving metadata for efficient loading
"""

import os
import sys
import argparse
import json
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def process_image(args):
    """Process a single image and its annotation."""
    img_path, anno_path, output_dir, img_size = args
    filename = os.path.basename(img_path)
    img_id = filename.split('.')[0]
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    try:
        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.LANCZOS)
        
        # Load and resize annotation
        anno = Image.open(anno_path)
        anno = anno.resize((img_size, img_size), Image.NEAREST)  # Use nearest for label maps
        
        # Save processed files
        out_img_path = os.path.join(output_dir, 'images', filename)
        out_anno_path = os.path.join(output_dir, 'annotations', filename)
        
        img.save(out_img_path)
        anno.save(out_anno_path)
        
        return img_id, True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return img_id, False

def main(args):
    # Check if COCO-Stuff dataset is downloaded
    stuff_dir = os.path.join(args.data_root, "cocostuff10k-master")
    if not os.path.exists(stuff_dir):
        print("COCO-Stuff dataset not found. Please run download_coco.py first.")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
    
    # Load image paths
    images_dir = os.path.join(stuff_dir, "dataset", "images")
    annotations_dir = os.path.join(stuff_dir, "dataset", "annotations")
    
    image_files = sorted(os.listdir(images_dir))
    
    # Create train/val/test splits
    np.random.seed(args.seed)
    np.random.shuffle(image_files)
    
    # Use predefined split ratios
    train_size = int(len(image_files) * args.train_ratio)
    val_size = int(len(image_files) * args.val_ratio)
    
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    split_map = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Save split information
    with open(os.path.join(args.output_dir, 'splits.json'), 'w') as f:
        json.dump({
            'train': [f.split('.')[0] for f in train_files],
            'val': [f.split('.')[0] for f in val_files],
            'test': [f.split('.')[0] for f in test_files]
        }, f, indent=2)
    
    # Process each split
    for split, files in split_map.items():
        print(f"Processing {split} split ({len(files)} images)...")
        
        tasks = []
        for filename in files:
            img_path = os.path.join(images_dir, filename)
            anno_path = os.path.join(annotations_dir, filename)
            output_dir = os.path.join(args.output_dir, split)
            tasks.append((img_path, anno_path, output_dir, args.img_size))
        
        # Process in parallel
        processed = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_image, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split}"):
                img_id, success = future.result()
                if success:
                    processed += 1
        
        print(f"Processed {processed}/{len(files)} images for {split} split")
    
    # Copy label map
    shutil.copy(
        os.path.join(args.data_root, 'stuff_label_map.json'),
        os.path.join(args.output_dir, 'label_map.json')
    )
    
    print("Dataset preparation complete!")
    print(f"Processed dataset is available at: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare COCO-Stuff dataset for inpainting")
    parser.add_argument('--data-root', type=str, default='./data/coco', 
                        help='Path to the downloaded dataset')
    parser.add_argument('--output-dir', type=str, default='./data/coco-processed', 
                        help='Path to store the processed dataset')
    parser.add_argument('--img-size', type=int, default=512, 
                        help='Size to resize images to')
    parser.add_argument('--train-ratio', type=float, default=0.8, 
                        help='Ratio of images to use for training')
    parser.add_argument('--val-ratio', type=float, default=0.1, 
                        help='Ratio of images to use for validation')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=4, 
                        help='Number of parallel workers')
    args = parser.parse_args()
    
    main(args) 