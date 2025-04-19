#!/usr/bin/env python3
"""
Training script for the comprehensive image inpainting model.
Handles data loading, training, validation, and checkpoint management.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import yaml
from pathlib import Path

from models.inpainting import ComprehensiveInpaintingModel
from datasets.inpainting_dataset import InpaintingDataset
from utils.visualization import save_visualization
from utils.logger import setup_logger
from utils.device import get_device
from utils.checkpoint import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Train Comprehensive Inpainting Model')
    
    # Basic arguments
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_root', type=str, help='Path to dataset root')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Device parameters
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (cuda, mps, cpu). If None, will automatically detect.')
    parser.add_argument('--disable_mps', action='store_true', help='Disable MPS even if available')
    
    # Data parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for DataLoader')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval for training')
    parser.add_argument('--save_interval', type=int, default=5000, help='Saving interval for checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5000, help='Evaluation interval')
    parser.add_argument('--vis_interval', type=int, default=1000, help='Visualization interval')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true', help='Debug mode (smaller dataset)')

    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_dataloaders(config, args):
    """Create training and validation data loaders."""
    data_root = args.data_root or config.get('data_root')
    if not data_root:
        raise ValueError("Data root must be specified in config or command line arguments.")
    
    # Get dataset configurations
    train_config = config.get('dataset', {}).get('train', {})
    val_config = config.get('dataset', {}).get('val', {})
    
    # If debug mode, use a very small subset
    if args.debug:
        train_config['max_samples'] = 100
        val_config['max_samples'] = 20
        
    # Create datasets
    train_dataset = InpaintingDataset(
        root=data_root,
        split='train',
        **train_config
    )
    
    val_dataset = InpaintingDataset(
        root=data_root,
        split='val',
        **val_config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, device, epoch, args, config, logger, global_step):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Training step
        stats = model.train_step(batch, optimizer, global_step, args.epochs * len(train_loader))
        
        # Update progress and global step
        total_loss += stats['total_loss']
        global_step += 1
        
        # Update progress bar
        pbar.set_postfix({"loss": stats['total_loss']})
        
        # Log losses
        if global_step % args.log_interval == 0:
            for k, v in stats.items():
                logger.info(f"Step {global_step}, {k}: {v:.4f}")
        
        # Save visualization
        if global_step % args.vis_interval == 0:
            vis_path = os.path.join(args.output_dir, 'visualizations', f'step_{global_step:07d}.png')
            with torch.no_grad():
                results = model(batch)
            save_visualization(batch, results, vis_path)
            logger.info(f"Saved visualization to {vis_path}")
        
        # Save checkpoint
        if global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, 'checkpoints', f'model_step_{global_step:07d}.pth')
            save_checkpoint(save_path, model, optimizer, epoch, global_step, 0)
            logger.info(f"Saved checkpoint to {save_path}")
        
        # Run validation
        if global_step % args.eval_interval == 0:
            validate(model, val_loader, device, global_step, args, config, logger)
    
    # Return average loss for the epoch
    epoch_loss = total_loss / len(train_loader)
    return epoch_loss, global_step

def validate(model, val_loader, device, global_step, args, config, logger):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0
    
    pbar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            results = model(batch)
            
            # Calculate losses
            losses = model._compute_losses(batch, results)
            total_loss += sum(losses.values())
            
            # Save example visualizations (first batch only)
            if batch_idx == 0:
                vis_path = os.path.join(args.output_dir, 'visualizations', f'val_step_{global_step:07d}.png')
                save_visualization(batch, results, vis_path)
    
    # Calculate average validation loss
    val_loss = total_loss / len(val_loader)
    logger.info(f"Validation loss: {val_loss:.4f}")
    
    return val_loss

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Set up logger
    logger = setup_logger('inpainting', os.path.join(args.output_dir, 'logs', 'train.log'))
    logger.info(f"Arguments: {args}")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Save configuration to output directory
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Get device
    device = get_device(args.device, disable_mps=args.disable_mps)
    logger.info(f"Using device: {device}")
    
    # Create model
    model = ComprehensiveInpaintingModel(config)
    model = model.to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # Load checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint_data = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint_data['epoch']
        global_step = checkpoint_data['step']
        logger.info(f"Resumed training from checkpoint {args.resume} (epoch {start_epoch}, step {global_step})")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(config, args)
    logger.info(f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Starting epoch {epoch}/{args.epochs}")
        
        # Train for one epoch
        epoch_loss, global_step = train_epoch(
            model, train_loader, optimizer, device, epoch, args, config, logger, global_step)
        
        # Log epoch results
        logger.info(f"Epoch {epoch} complete. Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        save_path = os.path.join(args.output_dir, 'checkpoints', f'model_epoch_{epoch:03d}.pth')
        save_checkpoint(save_path, model, optimizer, epoch, global_step, epoch_loss)
        logger.info(f"Saved epoch checkpoint to {save_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'checkpoints', 'model_final.pth')
    save_checkpoint(final_path, model, optimizer, args.epochs, global_step, epoch_loss)
    logger.info(f"Training completed. Saved final model to {final_path}")

if __name__ == "__main__":
    main() 