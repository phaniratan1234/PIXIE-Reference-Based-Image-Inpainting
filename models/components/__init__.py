"""
Inpainting Model Components

This package provides a collection of modular components for the inpainting architecture,
each responsible for specific aspects of the image inpainting process. These components
are designed to work together in a pipeline to achieve high-quality, coherent, and
contextually appropriate image completion.

The components follow a modular design pattern, allowing them to be:
1. Used together in a complete pipeline
2. Used individually for specific subtasks
3. Easily modified or replaced to test alternative approaches

The inpainting pipeline consists of:

1. ObjectDetector: Identifies and localizes objects in images, providing semantic
   understanding of the scene and object boundaries to guide inpainting.

2. SceneClassifier: Categorizes the scene type and content to ensure style and
   content consistency in the completed image.

3. StructuralAnalyzer: Detects and analyzes structural elements like edges,
   contours, and depth cues to maintain structural coherence.

4. UncertaintyEstimator: Predicts which areas of the inpainting process have
   higher uncertainty, allowing for refined focus during the generation process.

5. StructureBuilder: Creates the structural foundation of the inpainted region,
   ensuring proper perspective, object continuation, and spatial coherence.

6. TextureArtist: Generates detailed textures based on the structural foundation,
   producing the final visually coherent and detailed output.

Together, these components form a comprehensive pipeline that progressively builds
up the inpainted image from coarse structure to fine details, while maintaining
semantic and visual coherence with the surrounding content.
"""

from models.components.detection import ObjectDetector
from models.components.scene_classifier import SceneClassifier
from models.components.structural_analyzer import StructuralAnalyzer
from models.components.uncertainty_estimator import UncertaintyEstimator
from models.components.structure_builder import StructureBuilder
from models.components.texture_artist import TextureArtist

__all__ = [
    'ObjectDetector',
    'SceneClassifier',
    'StructuralAnalyzer',
    'UncertaintyEstimator',
    'StructureBuilder',
    'TextureArtist'
] 