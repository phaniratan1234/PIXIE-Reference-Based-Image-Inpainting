import os
import sys
import torch
import yaml
import argparse
from tqdm import tqdm
import traceback

# Make sure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    print("\n=== Testing Imports ===")
    
    try:
        import torch
        import numpy as np
        import yaml
        from PIL import Image
        import matplotlib.pyplot as plt
        
        # Project-specific imports
        from models.inpainting_model import InpaintingModel
        from models.components.detection import ObjectDetector
        from models.components.scene_classifier import SceneClassifier
        from models.components.structural_analyzer import StructuralAnalyzer
        from models.components.uncertainty_estimator import UncertaintyEstimator
        from models.components.structure_builder import StructureBuilder
        from models.components.texture_artist import TextureArtist
        from data.dataset import InpaintingDataset, build_dataloader
        
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        print(traceback.format_exc())
        return False

def test_model_creation():
    print("\n=== Testing Model Creation ===")
    
    try:
        from models.inpainting_model import InpaintingModel
        
        # Create minimal config for testing
        minimal_config = {
            'use_detection': False,
            'use_classification': False,
            'skip_uncertainty': False,
            'detection': {
                'pretrained': False,
                'num_classes': 80
            },
            'classification': {
                'pretrained': False,
                'num_classes': 182
            },
            'structural_analysis': {},
            'uncertainty': {},
            'structure_builder': {},
            'texture_artist': {}
        }
        
        model = InpaintingModel(minimal_config)
        print("✅ Model created successfully")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {num_params:,} parameters")
        
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        print(traceback.format_exc())
        return False

def test_mini_training_loop(data_root='data/coco-processed'):
    print("\n=== Testing Mini Training Loop ===")
    
    try:
        from models.inpainting_model import InpaintingModel
        from data.dataset import InpaintingDataset, build_dataloader
        
        if not os.path.exists(data_root):
            print(f"⚠️ Data directory {data_root} does not exist")
            
            # Create synthetic data
            print("Creating synthetic data for testing...")
            os.makedirs('test_data/train/images', exist_ok=True)
            os.makedirs('test_data/train/annotations', exist_ok=True)
            
            # Create a few test images
            for i in range(5):
                img = torch.rand(3, 256, 256).numpy() * 255
                img = img.transpose(1, 2, 0).astype('uint8')
                Image.fromarray(img).save(f'test_data/train/images/test_{i:03d}.jpg')
                
                # Create corresponding annotation (just a gray image)
                anno = np.ones((256, 256), dtype=np.uint8) * 128
                Image.fromarray(anno).save(f'test_data/train/annotations/test_{i:03d}.jpg')
            
            data_root = 'test_data'
        
        # Create minimal config
        config = {
            'data_root': data_root,
            'dataset': {
                'train': {'image_size': 256, 'mask_type': 'random', 'augmentation': True, 'max_samples': 5},
            },
            'batch_size': 2,
            'use_detection': False,
            'use_classification': False,
            'skip_uncertainty': False,
            'detection': {
                'pretrained': False,
                'num_classes': 80
            },
            'classification': {
                'pretrained': False,
                'num_classes': 182
            },
            'structural_analysis': {},
            'uncertainty': {},
            'structure_builder': {},
            'texture_artist': {}
        }
        
        # Create model
        model = InpaintingModel(config)
        model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # Create dataloader
        dataloader = build_dataloader(config, split='train', num_workers=0)
        
        print(f"Mini training loop with {len(dataloader)} batches")
        
        # Execute a mini training loop (just 2 steps)
        for i, batch in enumerate(dataloader):
            if i >= 2:
                break
                
            print(f"Processing batch {i+1}/2...")
            
            # Move data to CPU (for testing)
            batch = {k: v if not isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            results = model(
                batch['masked_image'], 
                batch['mask'], 
                batch['image']
            )
            
            # Check if loss is computed
            if 'total_loss' in results:
                loss = results['total_loss']
                print(f"Loss: {loss.item():.6f}")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print("✅ Backward pass successful")
            else:
                print("⚠️ No loss computed in results")
            
            print(f"✅ Batch {i+1} processed successfully")
        
        print("✅ Mini training loop completed successfully")
        return True
    except Exception as e:
        print(f"❌ Training loop error: {e}")
        print(traceback.format_exc())
        return False

def test_full_pipeline():
    print("=== Testing Full Pipeline ===")
    print("This will test the entire inpainting pipeline from data loading to model training.")
    
    success = test_imports()
    if not success:
        print("❌ Import tests failed, stopping pipeline test")
        return False
    
    success = test_model_creation()
    if not success:
        print("❌ Model creation failed, stopping pipeline test")
        return False
    
    # Try to find the dataset directory
    possible_data_roots = [
        'data/coco-processed',
        'data/coco_processed',
        'data/coco-stuff-processed',
        'data/coco_stuff_processed',
        'data/coco',
        'test_data'
    ]
    
    data_root = None
    for path in possible_data_roots:
        if os.path.exists(path):
            data_root = path
            break
    
    success = test_mini_training_loop(data_root)
    if not success:
        print("❌ Mini training loop failed")
        return False
    
    print("\n✅ Full pipeline test completed successfully")
    print("You can now proceed with the full training using:")
    print("python train.py --config config/default.yaml --data_root data/coco-processed --output_dir outputs")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the inpainting pipeline")
    parser.add_argument('--data_root', type=str, default=None, help='Path to dataset root directory')
    args = parser.parse_args()
    
    if args.data_root:
        test_mini_training_loop(args.data_root)
    else:
        test_full_pipeline() 