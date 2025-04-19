import torch
import torch.nn as nn
import traceback
import gc
import os
import sys
from models.components.detection import ObjectDetector
from models.components.scene_classifier import SceneClassifier
from models.components.structural_analyzer import StructuralAnalyzer
from models.components.uncertainty_estimator import UncertaintyEstimator
from models.components.structure_builder import StructureBuilder
from models.components.texture_artist import TextureArtist

def test_component(component_name):
    """
    Test a single component by name to avoid memory issues
    """
    print(f"\n=== Testing {component_name} ===")
    
    # Force CPU execution for testing to avoid GPU memory issues
    device = torch.device('cpu')
    
    # Create tiny input data
    batch_size = 1
    h, w = 64, 64
    images = torch.rand(batch_size, 3, h, w, device=device)
    masks = torch.ones(batch_size, 1, h, w, device=device)
    masks[:, :, h//4:h//2, w//4:w//2] = 0  # Create masked region
    
    try:
        if component_name == "ObjectDetector":
            component = ObjectDetector(pretrained=False, num_classes=10)
            component = component.to(device)
            with torch.no_grad():
                result = component(images, masks)
            print(f"Output keys: {list(result.keys())}")
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)}")
            print("✅ ObjectDetector passed!")
            
        elif component_name == "SceneClassifier":
            component = SceneClassifier(pretrained=False, num_classes=10)
            component = component.to(device)
            with torch.no_grad():
                result = component(images, masks)
            print(f"Output keys: {list(result.keys())}")
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)}")
            print("✅ SceneClassifier passed!")
            
        elif component_name == "StructuralAnalyzer":
            component = StructuralAnalyzer(edge_channels=8, structure_channels=16)
            component = component.to(device)
            with torch.no_grad():
                result = component(images, masks)
            print(f"Output keys: {list(result.keys())}")
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)}")
            print("✅ StructuralAnalyzer passed!")
            
        elif component_name == "UncertaintyEstimator":
            component = UncertaintyEstimator(in_channels=3, hidden_channels=16, uncertainty_types=1)
            component = component.to(device)
            with torch.no_grad():
                result = component(images, masks)
            print(f"Output keys: {list(result.keys())}")
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)}")
            print("✅ UncertaintyEstimator passed!")
            
        elif component_name == "StructureBuilder":
            component = StructureBuilder(
                in_channels=3, 
                hidden_channels=16, 
                attention_channels=16
            )
            component = component.to(device)
            
            # Create simple edge features
            edge_features = torch.rand(batch_size, 1, h, w, device=device)
            
            with torch.no_grad():
                result = component(images, edge_features=edge_features, masks=masks)
            print(f"Output keys: {list(result.keys())}")
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)}")
            print("✅ StructureBuilder passed!")
            
        elif component_name == "TextureArtist":
            # Set specific parameters to ensure compatibility
            in_channels = 8
            feature_channels = 8
            output_channels = 3
            component = TextureArtist(
                in_channels=in_channels, 
                feature_channels=feature_channels, 
                output_channels=output_channels, 
                use_spectral_norm=False
            )
            component = component.to(device)
            
            # Create structure features with the right dimensions
            structure_features = torch.rand(batch_size, in_channels, h, w, device=device)
            
            with torch.no_grad():
                result = component(structure_features, original_images=images, masks=masks)
            print(f"Output keys: {list(result.keys())}")
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)}")
            print("✅ TextureArtist passed!")
            
        else:
            print(f"❌ Unknown component: {component_name}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ {component_name} failed: {e}")
        print(traceback.format_exc())
        return False
    finally:
        # Clean up memory
        gc.collect()

if __name__ == "__main__":
    # Disable GPU usage for testing
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print("GPU disabled for testing")
    
    # Get component to test from command line or test all
    if len(sys.argv) > 1:
        component_name = sys.argv[1]
        test_component(component_name)
    else:
        # Test each component individually to avoid memory issues
        components = [
            "ObjectDetector",
            "SceneClassifier", 
            "StructuralAnalyzer",
            "UncertaintyEstimator",
            "StructureBuilder",
            "TextureArtist"
        ]
        
        passed = 0
        for component in components:
            result = test_component(component)
            if result:
                passed += 1
            # Clean up memory between tests
            gc.collect()
        
        print(f"\n=== Component Testing Summary ===")
        print(f"✅ {passed}/{len(components)} components passed") 