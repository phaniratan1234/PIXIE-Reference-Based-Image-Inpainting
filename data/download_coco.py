#!/usr/bin/env python3
"""
Script to download and prepare the COCO-Stuff 10K dataset for image inpainting.
"""

import os
import sys
import argparse
import zipfile
import shutil
import subprocess
from tqdm import tqdm
import gdown
import requests
import json

def download_file(url, destination, description=None):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    desc = description if description else f"Downloading {os.path.basename(destination)}"
    
    with open(destination, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

def main(args):
    # Create necessary directories
    os.makedirs(args.data_root, exist_ok=True)
    
    print("Downloading COCO-Stuff 10K dataset...")
    
    # Download COCO-Stuff 10K annotations
    stuff_url = "https://github.com/nightrome/cocostuff10k/archive/refs/heads/master.zip"
    stuff_zip = os.path.join(args.data_root, "cocostuff10k.zip")
    
    if not os.path.exists(stuff_zip):
        download_file(stuff_url, stuff_zip, "Downloading COCO-Stuff 10K")
    
    # Download COCO 2017 images
    if not os.path.exists(os.path.join(args.data_root, 'train2017.zip')):
        print("Downloading COCO 2017 train images...")
        train_url = "http://images.cocodataset.org/zips/train2017.zip"
        download_file(train_url, os.path.join(args.data_root, 'train2017.zip'), "Downloading COCO train2017")
    
    if not os.path.exists(os.path.join(args.data_root, 'val2017.zip')):
        print("Downloading COCO 2017 validation images...")
        val_url = "http://images.cocodataset.org/zips/val2017.zip"
        download_file(val_url, os.path.join(args.data_root, 'val2017.zip'), "Downloading COCO val2017")
    
    if not os.path.exists(os.path.join(args.data_root, 'annotations_trainval2017.zip')):
        print("Downloading COCO 2017 annotations...")
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        download_file(annotations_url, os.path.join(args.data_root, 'annotations_trainval2017.zip'), "Downloading COCO annotations")
    
    # Extract the datasets
    print("Extracting COCO-Stuff 10K...")
    with zipfile.ZipFile(stuff_zip, 'r') as zip_ref:
        zip_ref.extractall(args.data_root)
    
    print("Extracting COCO 2017 train images...")
    with zipfile.ZipFile(os.path.join(args.data_root, 'train2017.zip'), 'r') as zip_ref:
        zip_ref.extractall(args.data_root)
    
    print("Extracting COCO 2017 validation images...")
    with zipfile.ZipFile(os.path.join(args.data_root, 'val2017.zip'), 'r') as zip_ref:
        zip_ref.extractall(args.data_root)
    
    print("Extracting COCO 2017 annotations...")
    with zipfile.ZipFile(os.path.join(args.data_root, 'annotations_trainval2017.zip'), 'r') as zip_ref:
        zip_ref.extractall(args.data_root)
    
    # Create a mapping file for COCO-Stuff category labels
    stuff_dir = os.path.join(args.data_root, "cocostuff10k-master")
    categories_json = os.path.join(stuff_dir, "dataset", "cocostuff-10k-v1.1.json")
    
    with open(categories_json, 'r') as f:
        data = json.load(f)
    
    # Create a simplified label map
    label_map = {}
    for category in data['categories']:
        label_map[category['id']] = {
            'name': category['name'],
            'supercategory': category['supercategory']
        }
    
    # Save the label map
    with open(os.path.join(args.data_root, 'stuff_label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print("Dataset download and preparation complete!")
    print(f"Data is available at: {args.data_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare COCO-Stuff dataset")
    parser.add_argument('--data-root', type=str, default='./data/coco', 
                        help='Path to store the dataset')
    args = parser.parse_args()
    
    main(args) 