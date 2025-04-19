import torch
import yaml
import os
import traceback
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from data.dataset import InpaintingDataset, build_dataloader

def test_dataset(data_root='data/coco-processed'):
    print("\n=== Testing Dataset ===")
    
    if not os.path.exists(data_root):
        print(f"⚠️ Data directory {data_root} does not exist, skipping dataset test")
        return False
    
    try:
        # Check if it's the minimal processed structure
        if os.path.exists(os.path.join(data_root, 'train', 'images')):
            print(f"Found COCO-Stuff processed directory structure")
            
            # Test with random masks first
            dataset = InpaintingDataset(
                data_root=data_root,
                split='train',
                image_size=256,
                mask_type='random',
                max_samples=5
            )
            
            print(f"✅ Dataset initialized with {len(dataset)} samples")
            
            # Test getting a sample
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Mask shape: {sample['mask'].shape}")
            
            # Test different mask types
            for mask_type in ['random', 'bbox', 'segmentation']:
                print(f"\nTesting mask type '{mask_type}'...")
                try:
                    dataset.mask_type = mask_type
                    sample = dataset[0]
                    print(f"Mask min: {sample['mask'].min().item()}, max: {sample['mask'].max().item()}")
                    # Calculate masked percentage (percentage of zeros)
                    masked_percentage = (sample['mask'] == 0).float().mean().item() * 100
                    print(f"Masked area: {masked_percentage:.2f}%")
                    print(f"✅ Mask type '{mask_type}' works")
                except Exception as e:
                    print(f"❌ Mask type '{mask_type}' failed: {e}")
                    print(traceback.format_exc())
            
            return True
        else:
            print(f"⚠️ COCO-Stuff processed directory structure not found in {data_root}")
            return False
            
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        print(traceback.format_exc())
        return False

def test_dataloader(data_root='data/coco-processed'):
    print("\n=== Testing DataLoader ===")
    
    if not os.path.exists(data_root):
        print(f"⚠️ Data directory {data_root} does not exist, skipping dataloader test")
        return False
    
    try:
        # Create a minimal config for testing
        config = {
            'data_root': data_root,
            'dataset': {
                'train': {'image_size': 256, 'mask_type': 'random', 'augmentation': True, 'max_samples': 5},
                'val': {'image_size': 256, 'mask_type': 'random', 'augmentation': False, 'max_samples': 5}
            },
            'batch_size': 2
        }
        
        # Build dataloader
        dataloader = build_dataloader(config, split='train', num_workers=0)
        
        print(f"✅ DataLoader initialized with {len(dataloader)} batches")
        
        # Test getting a batch
        batch = next(iter(dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch mask shape: {batch['mask'].shape}")
        
        # Display a sample if running in a notebook (commented out for regular script)
        # plt.figure(figsize=(12, 4))
        # plt.subplot(131)
        # plt.imshow(batch['image'][0].permute(1,2,0).clip(0, 1))
        # plt.title('Original')
        # plt.subplot(132)
        # plt.imshow(batch['mask'][0].permute(1,2,0).squeeze(), cmap='gray')
        # plt.title('Mask')
        # plt.subplot(133)
        # plt.imshow(batch['masked_image'][0].permute(1,2,0).clip(0, 1))
        # plt.title('Masked')
        # plt.tight_layout()
        # plt.show()
        
        return True
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        print(traceback.format_exc())
        return False

def test_with_sample_image():
    print("\n=== Testing with Sample Image ===")
    
    try:
        # Create a synthetic test image and mask
        h, w = 512, 512
        test_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw a red rectangle
        test_img[100:200, 100:200, 0] = 255
        
        # Draw a green circle
        center = (300, 300)
        radius = 80
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        circle = dist <= radius
        test_img[circle, 1] = 255
        
        # Draw a blue triangle
        pts = np.array([[400, 100], [500, 100], [450, 200]])
        cv_img = test_img.copy()
        try:
            import cv2
            cv2.fillPoly(cv_img, [pts], (0, 0, 255))
            test_img = cv_img
        except ImportError:
            # If OpenCV is not available, just use the rectangle and circle
            pass
        
        # Create a test mask (0 for masked region, 1 for valid region)
        test_mask = np.ones((h, w), dtype=np.uint8)
        test_mask[200:350, 200:350] = 0  # Mask the center region
        
        # Save the test image and mask
        os.makedirs('test_data', exist_ok=True)
        test_img_path = 'test_data/test_image.png'
        test_mask_path = 'test_data/test_mask.png'
        
        try:
            from PIL import Image
            Image.fromarray(test_img).save(test_img_path)
            Image.fromarray(test_mask * 255).save(test_mask_path)
            print(f"✅ Created test image and mask in test_data/")
        except Exception as e:
            print(f"❌ Failed to save test files: {e}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Test with sample image failed: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("=== Testing Data Pipeline ===")
    print("Note: These tests will be skipped if data directories don't exist")
    
    # Try to find the dataset directory
    possible_data_roots = [
        'data/coco-processed',
        'data/coco_processed',
        'data/coco-stuff-processed',
        'data/coco_stuff_processed',
        'data/coco'
    ]
    
    data_root = None
    for path in possible_data_roots:
        if os.path.exists(path):
            data_root = path
            break
    
    if data_root:
        print(f"Found data directory: {data_root}")
        test_dataset(data_root)
        test_dataloader(data_root)
    else:
        print("No data directory found. Generating sample test data.")
        test_with_sample_image() 