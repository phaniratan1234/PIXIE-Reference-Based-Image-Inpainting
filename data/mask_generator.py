#!/usr/bin/env python3
"""
Advanced mask generation techniques for image inpainting training.
Implements various mask types:
- Object-aware masks (remove entire objects)
- Boundary-crossing masks (cross object boundaries)
- Damage-simulation masks (scratches, tears, etc.)
- Irregular masks (random shapes)
"""

import numpy as np
import cv2
import random
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, label
import torch
from skimage.draw import line, circle, bezier_curve
from skimage.filters import gaussian
from scipy.spatial import Voronoi, voronoi_plot_2d

class MaskGenerator:
    """Base class for mask generation."""
    
    def __init__(self, height=512, width=512, mask_color=1):
        """
        Initialize the mask generator.
        
        Args:
            height (int): Height of the mask
            width (int): Width of the mask
            mask_color (int): Value to use for the masked regions
        """
        self.height = height
        self.width = width
        self.mask_color = mask_color
    
    def generate(self):
        """Generate a binary mask."""
        raise NotImplementedError("Subclasses must implement this method")

class RandomIrregularMaskGenerator(MaskGenerator):
    """Generate irregular masks using random brushes and lines."""
    
    def __init__(self, height=512, width=512, num_vertices=(4, 12), 
                 max_angle=2*np.pi, min_size=0.1, max_size=0.5, 
                 brush_width=(10, 40), mask_color=1):
        """
        Initialize the irregular mask generator.
        
        Args:
            height (int): Height of the mask
            width (int): Width of the mask
            num_vertices (tuple): Range of number of vertices for random polygons
            max_angle (float): Maximum angle for irregular polygon vertices
            min_size (float): Minimum size of mask as fraction of image size
            max_size (float): Maximum size of mask as fraction of image size
            brush_width (tuple): Range of brush widths for strokes
            mask_color (int): Value to use for the masked regions
        """
        super().__init__(height, width, mask_color)
        self.num_vertices = num_vertices
        self.max_angle = max_angle
        self.min_size = min_size
        self.max_size = max_size
        self.brush_width = brush_width
    
    def generate(self):
        """Generate an irregular mask using random brushes and lines."""
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Determine mask size
        size = np.random.uniform(self.min_size, self.max_size) * self.width
        
        # Random number of vertices
        num_vertices = np.random.randint(self.num_vertices[0], self.num_vertices[1] + 1)
        
        # Generate random vertices for polygon
        angle_step = 2 * np.pi / num_vertices
        angles = np.random.uniform(0, self.max_angle, size=num_vertices)
        
        # Random center point
        center_x = np.random.randint(self.width // 4, self.width * 3 // 4)
        center_y = np.random.randint(self.height // 4, self.height * 3 // 4)
        
        # Generate vertices
        vertices = []
        for i in range(num_vertices):
            angle = i * angle_step + angles[i]
            radius = size / 2 * np.random.uniform(0.8, 1.2)
            
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            x = min(max(x, 0), self.width - 1)
            y = min(max(y, 0), self.height - 1)
            
            vertices.append((x, y))
        
        # Draw irregular polygon
        cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], self.mask_color)
        
        # Add some random brushes
        num_brushes = np.random.randint(1, 4)
        for _ in range(num_brushes):
            brush_width = np.random.randint(self.brush_width[0], self.brush_width[1])
            x0, y0 = random.choice(vertices)
            x1, y1 = random.choice(vertices)
            
            cv2.line(mask, (x0, y0), (x1, y1), self.mask_color, brush_width)
        
        return mask.astype(np.float32)

class ObjectAwareMaskGenerator(MaskGenerator):
    """Generate masks that target complete objects in a segmentation map."""
    
    def __init__(self, height=512, width=512, dilation_kernel_size=15, mask_color=1):
        """
        Initialize the object-aware mask generator.
        
        Args:
            height (int): Height of the mask
            width (int): Width of the mask
            dilation_kernel_size (int): Size of kernel for dilating the mask
            mask_color (int): Value to use for the masked regions
        """
        super().__init__(height, width, mask_color)
        self.dilation_kernel_size = dilation_kernel_size
        self.kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    
    def generate(self, segmentation_map):
        """
        Generate an object-aware mask.
        
        Args:
            segmentation_map (numpy.ndarray): Segmentation map with object IDs
            
        Returns:
            numpy.ndarray: Binary mask
        """
        # Ensure segmentation map is properly sized
        segmentation_map = cv2.resize(segmentation_map, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Identify unique objects
        unique_objects = np.unique(segmentation_map)
        
        # Remove background (typically label 0)
        if 0 in unique_objects:
            unique_objects = unique_objects[unique_objects != 0]
        
        # No objects found
        if len(unique_objects) == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        # Randomly select 1-3 objects to mask
        num_objects = min(len(unique_objects), np.random.randint(1, 4))
        selected_objects = np.random.choice(unique_objects, size=num_objects, replace=False)
        
        # Create mask for selected objects
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        for obj_id in selected_objects:
            object_mask = (segmentation_map == obj_id).astype(np.uint8)
            
            # Dilate the mask to ensure complete coverage
            object_mask = cv2.dilate(object_mask, self.kernel, iterations=1)
            
            # Add to overall mask
            mask = np.maximum(mask, object_mask)
        
        # Ensure mask is valid
        mask = mask * self.mask_color
        
        # Add some perlin noise to the edges for more natural transition
        mask = self._add_edge_noise(mask)
        
        return mask.astype(np.float32)
    
    def _add_edge_noise(self, mask):
        """Add noise to mask edges to create more natural transitions."""
        # Find edges
        edges = cv2.Canny(mask.astype(np.uint8), 100, 200)
        
        # Dilate edges
        edge_zone = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
        
        # Create perlin-like noise
        noise = np.random.uniform(0, 1, (self.height, self.width))
        noise = gaussian(noise, sigma=3)
        
        # Normalize noise to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Apply noise only on edge zones
        edge_noise = edge_zone / 255.0 * noise * 0.5
        
        # Apply edge noise - expand mask in some areas, shrink in others
        adjusted_mask = np.clip(mask - edge_noise + (edge_zone / 255.0) * 0.3, 0, 1)
        
        return adjusted_mask

class BoundaryCrossingMaskGenerator(MaskGenerator):
    """Generate masks that intentionally cross object boundaries."""
    
    def __init__(self, height=512, width=512, min_size=0.1, max_size=0.3, mask_color=1):
        """
        Initialize the boundary-crossing mask generator.
        
        Args:
            height (int): Height of the mask
            width (int): Width of the mask
            min_size (float): Minimum size of mask as fraction of image size
            max_size (float): Maximum size of mask as fraction of image size
            mask_color (int): Value to use for the masked regions
        """
        super().__init__(height, width, mask_color)
        self.min_size = min_size
        self.max_size = max_size
    
    def generate(self, segmentation_map):
        """
        Generate a boundary-crossing mask.
        
        Args:
            segmentation_map (numpy.ndarray): Segmentation map with object IDs
            
        Returns:
            numpy.ndarray: Binary mask
        """
        # Ensure segmentation map is properly sized
        segmentation_map = cv2.resize(segmentation_map, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Identify boundaries
        gradient_x = cv2.Sobel(segmentation_map, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(segmentation_map, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Threshold to get boundary pixels
        boundaries = gradient_magnitude > 0
        
        # If no boundaries found, fall back to random mask
        if not np.any(boundaries):
            generator = RandomIrregularMaskGenerator(self.height, self.width, mask_color=self.mask_color)
            return generator.generate()
        
        # Find boundary coordinates
        boundary_coords = np.column_stack(np.where(boundaries))
        
        if len(boundary_coords) == 0:
            generator = RandomIrregularMaskGenerator(self.height, self.width, mask_color=self.mask_color)
            return generator.generate()
        
        # Randomly select a boundary point as center of mask
        center_idx = np.random.randint(0, len(boundary_coords))
        center_y, center_x = boundary_coords[center_idx]
        
        # Create circular mask with random radius
        size = np.random.uniform(self.min_size, self.max_size) * min(self.height, self.width)
        Y, X = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Create basic circular mask
        mask = dist_from_center <= size / 2
        
        # Make mask irregular by adding randomness to the radius
        angles = np.linspace(0, 2*np.pi, 30)
        radii = size/2 * np.random.uniform(0.8, 1.2, size=30)
        
        # Create vertices for irregular polygon
        vertices = []
        for angle, radius in zip(angles, radii):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            x = min(max(x, 0), self.width - 1)
            y = min(max(y, 0), self.height - 1)
            
            vertices.append((x, y))
        
        # Convert to numpy for OpenCV
        vertices = np.array(vertices, np.int32).reshape((-1, 1, 2))
        
        # Create empty mask and draw polygon
        irregular_mask = np.zeros((self.height, self.width), dtype=np.float32)
        cv2.fillPoly(irregular_mask, [vertices], self.mask_color)
        
        # Validate coverage of multiple objects
        if not self._validate_boundary_crossing(irregular_mask, segmentation_map):
            # If validation fails, retry with different shape
            return self.generate(segmentation_map)
        
        return irregular_mask.astype(np.float32)
    
    def _validate_boundary_crossing(self, mask, segmentation_map, min_objects=2, min_coverage=0.3):
        """Validate that the mask crosses boundaries of multiple objects."""
        # Get unique objects in the masked region
        masked_region = mask > 0
        masked_segments = segmentation_map[masked_region]
        unique_objects = np.unique(masked_segments)
        
        # Count objects with sufficient coverage
        valid_objects = 0
        for obj_id in unique_objects:
            # Skip background
            if obj_id == 0:
                continue
            
            # Calculate coverage
            obj_mask = (segmentation_map == obj_id)
            obj_masked = np.logical_and(obj_mask, masked_region)
            coverage = np.sum(obj_masked) / np.sum(obj_mask)
            
            if coverage >= min_coverage:
                valid_objects += 1
        
        return valid_objects >= min_objects

class DamageSimulationMaskGenerator(MaskGenerator):
    """Generate masks that simulate real-world damage like scratches, tears, etc."""
    
    def __init__(self, height=512, width=512, damage_type='mixed', mask_color=1):
        """
        Initialize the damage simulation mask generator.
        
        Args:
            height (int): Height of the mask
            width (int): Width of the mask
            damage_type (str): Type of damage to simulate ('scratch', 'tear', 'water', 'mixed')
            mask_color (int): Value to use for the masked regions
        """
        super().__init__(height, width, mask_color)
        self.damage_type = damage_type
        self.damage_generators = {
            'scratch': self._generate_scratch,
            'tear': self._generate_tear,
            'water': self._generate_water_damage,
            'mixed': self._generate_mixed_damage
        }
    
    def generate(self):
        """Generate a damage simulation mask."""
        # Choose generator based on damage_type
        if self.damage_type == 'mixed':
            # Randomly choose a damage type for mixed mode
            damage_type = random.choice(['scratch', 'tear', 'water'])
            generator = self.damage_generators[damage_type]
        else:
            generator = self.damage_generators[self.damage_type]
        
        mask = generator()
        return mask.astype(np.float32)
    
    def _generate_scratch(self):
        """Generate scratch-like damage."""
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Number of scratches
        num_scratches = np.random.randint(1, 5)
        
        for _ in range(num_scratches):
            # Random starting point near edge
            edge_type = np.random.randint(0, 4)
            
            if edge_type == 0:  # Top edge
                start_x = np.random.randint(0, self.width)
                start_y = 0
            elif edge_type == 1:  # Right edge
                start_x = self.width - 1
                start_y = np.random.randint(0, self.height)
            elif edge_type == 2:  # Bottom edge
                start_x = np.random.randint(0, self.width)
                start_y = self.height - 1
            else:  # Left edge
                start_x = 0
                start_y = np.random.randint(0, self.height)
            
            # Random end point within image
            end_x = np.random.randint(0, self.width)
            end_y = np.random.randint(0, self.height)
            
            # Number of control points for Bezier curve
            num_control_points = np.random.randint(2, 5)
            control_points = []
            
            # Generate random control points
            for i in range(num_control_points):
                ctrl_x = start_x + (end_x - start_x) * (i + 1) / (num_control_points + 1)
                ctrl_y = start_y + (end_y - start_y) * (i + 1) / (num_control_points + 1)
                
                # Add random deviation
                ctrl_x += np.random.randint(-self.width // 10, self.width // 10)
                ctrl_y += np.random.randint(-self.height // 10, self.height // 10)
                
                control_points.append((ctrl_x, ctrl_y))
            
            # Create points for bezier curve
            points = [(start_x, start_y)] + control_points + [(end_x, end_y)]
            points = np.array(points)
            
            # Generate bezier curve
            rr, cc = bezier_curve(points[:, 1], points[:, 0], 100)
            
            # Keep points inside the image
            valid_indices = (rr >= 0) & (rr < self.height) & (cc >= 0) & (cc < self.width)
            rr, cc = rr[valid_indices], cc[valid_indices]
            
            # Random width variation
            widths = np.random.randint(2, 10, size=len(rr))
            
            # Draw scratch
            for i, (r, c) in enumerate(zip(rr, cc)):
                cv2.circle(mask, (int(c), int(r)), int(widths[i]), self.mask_color, -1)
        
        return mask
    
    def _generate_tear(self):
        """Generate tear-like damage."""
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Start with a straight line
        edge_type = np.random.randint(0, 4)
        
        if edge_type == 0:  # Horizontal tear
            y = np.random.randint(self.height // 4, self.height * 3 // 4)
            points = [(0, y), (self.width - 1, y)]
        else:  # Vertical tear
            x = np.random.randint(self.width // 4, self.width * 3 // 4)
            points = [(x, 0), (x, self.height - 1)]
        
        # Add jagged deviations
        num_points = np.random.randint(5, 15)
        x_points = np.linspace(points[0][0], points[1][0], num_points)
        y_points = np.linspace(points[0][1], points[1][1], num_points)
        
        # Add random deviations to make it jagged
        jagged_points = []
        for x, y in zip(x_points, y_points):
            # Add random deviation perpendicular to the tear direction
            if edge_type == 0:  # Horizontal tear
                dev = np.random.randint(-self.height // 20, self.height // 20)
                jagged_points.append((int(x), int(y + dev)))
            else:  # Vertical tear
                dev = np.random.randint(-self.width // 20, self.width // 20)
                jagged_points.append((int(x + dev), int(y)))
        
        # Draw tear
        thickness = np.random.randint(5, 15)
        for i in range(len(jagged_points) - 1):
            cv2.line(mask, jagged_points[i], jagged_points[i + 1], self.mask_color, thickness)
        
        # Add some branching with probability
        branch_prob = 0.7
        if np.random.rand() < branch_prob:
            branch_point_idx = np.random.randint(1, len(jagged_points) - 1)
            branch_start = jagged_points[branch_point_idx]
            
            # Branch end point
            if edge_type == 0:  # Branch from horizontal tear
                branch_end_x = branch_start[0] + np.random.randint(-self.width // 4, self.width // 4)
                branch_dir = np.random.choice([-1, 1])  # Up or down
                branch_end_y = branch_start[1] + branch_dir * np.random.randint(self.height // 8, self.height // 3)
            else:  # Branch from vertical tear
                branch_dir = np.random.choice([-1, 1])  # Left or right
                branch_end_x = branch_start[0] + branch_dir * np.random.randint(self.width // 8, self.width // 3)
                branch_end_y = branch_start[1] + np.random.randint(-self.height // 4, self.height // 4)
            
            # Keep branch within image
            branch_end_x = min(max(branch_end_x, 0), self.width - 1)
            branch_end_y = min(max(branch_end_y, 0), self.height - 1)
            
            # Draw branch
            branch_thickness = max(1, thickness - 2)
            cv2.line(mask, branch_start, (branch_end_x, branch_end_y), self.mask_color, branch_thickness)
        
        return mask
    
    def _generate_water_damage(self):
        """Generate water damage-like mask using Voronoi tessellation."""
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Number of "droplets"
        num_droplets = np.random.randint(10, 30)
        
        # Generate random points for Voronoi tessellation
        points = np.random.rand(num_droplets, 2)
        points[:, 0] *= self.width
        points[:, 1] *= self.height
        
        # Create Voronoi diagram
        vor = Voronoi(points)
        
        # Create mask from random regions
        num_regions = np.random.randint(1, 5)
        selected_regions = np.random.choice(len(vor.regions), num_regions, replace=False)
        
        # Create PIL image and draw
        pil_mask = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(pil_mask)
        
        for region_idx in selected_regions:
            region = vor.regions[region_idx]
            if not -1 in region and len(region) > 0:
                # Get vertices of the region
                polygon = [tuple(vor.vertices[i]) for i in region]
                
                # Filter out vertices outside the image
                polygon = [p for p in polygon if 0 <= p[0] < self.width and 0 <= p[1] < self.height]
                
                if len(polygon) >= 3:
                    # Draw the filled polygon
                    draw.polygon(polygon, fill=int(self.mask_color * 255))
        
        # Convert back to numpy
        mask = np.array(pil_mask) / 255.0 * self.mask_color
        
        # Apply Gaussian blur for softer edges
        mask = gaussian(mask, sigma=5)
        
        # Threshold to make it binary
        mask = (mask > 0.5 * self.mask_color).astype(np.float32) * self.mask_color
        
        # Simulate gravity effects and flow patterns
        mask = self._add_gravity_effect(mask)
        
        return mask
    
    def _add_gravity_effect(self, mask):
        """Add gravity effects to water damage mask."""
        # Find the bottom edge of the water spots
        mask_binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # For each contour, add drips
        for contour in contours:
            # Find the bottom-most point
            bottom_point = None
            max_y = -1
            
            for point in contour:
                x, y = point[0]
                if y > max_y:
                    max_y = y
                    bottom_point = (x, y)
            
            if bottom_point and max_y < self.height - 10:
                # Add random drips
                num_drips = np.random.randint(0, 3)
                
                for _ in range(num_drips):
                    x_offset = np.random.randint(-10, 11)
                    drip_x = min(max(bottom_point[0] + x_offset, 0), self.width - 1)
                    drip_length = np.random.randint(10, 50)
                    drip_end_y = min(bottom_point[1] + drip_length, self.height - 1)
                    
                    # Create a wavy drip line
                    x_wave = np.sin(np.linspace(0, np.pi, drip_length)) * 5
                    
                    for i, y in enumerate(range(bottom_point[1], drip_end_y)):
                        if i < len(x_wave):
                            x = int(drip_x + x_wave[i])
                            x = min(max(x, 0), self.width - 1)
                            
                            # Make drip thinner as it gets longer
                            thickness = max(1, int(5 * (1 - i / drip_length)))
                            cv2.circle(mask, (x, y), thickness, self.mask_color, -1)
        
        return mask
    
    def _generate_mixed_damage(self):
        """Generate a mix of different damage types."""
        # Randomly select 2-3 damage types
        num_damage_types = np.random.randint(2, 4)
        damage_types = np.random.choice(['scratch', 'tear', 'water'], num_damage_types, replace=False)
        
        # Combine masks
        combined_mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        for damage_type in damage_types:
            generator = self.damage_generators[damage_type]
            mask = generator()
            combined_mask = np.maximum(combined_mask, mask)
        
        return combined_mask

class MixedMaskGenerator(MaskGenerator):
    """Generate masks using a mix of different techniques."""
    
    def __init__(self, height=512, width=512, mask_color=1,
                 irregular_prob=0.6, boundary_prob=0.2, object_prob=0.3, damage_prob=0.3,
                 min_size=0.1, max_size=0.5):
        """
        Initialize the mixed mask generator.
        
        Args:
            height (int): Height of the mask
            width (int): Width of the mask
            mask_color (int): Value to use for the masked regions
            irregular_prob (float): Probability of using irregular masks
            boundary_prob (float): Probability of using boundary-crossing masks
            object_prob (float): Probability of using object-aware masks
            damage_prob (float): Probability of using damage-simulation masks
            min_size (float): Minimum size of mask as fraction of image size
            max_size (float): Maximum size of mask as fraction of image size
        """
        super().__init__(height, width, mask_color)
        self.irregular_prob = irregular_prob
        self.boundary_prob = boundary_prob
        self.object_prob = object_prob
        self.damage_prob = damage_prob
        self.min_size = min_size
        self.max_size = max_size
        
        # Initialize generators
        self.irregular_generator = RandomIrregularMaskGenerator(
            height, width, min_size=min_size, max_size=max_size, mask_color=mask_color
        )
        self.damage_generator = DamageSimulationMaskGenerator(
            height, width, mask_color=mask_color
        )
        self.object_generator = ObjectAwareMaskGenerator(
            height, width, mask_color=mask_color
        )
        self.boundary_generator = BoundaryCrossingMaskGenerator(
            height, width, min_size=min_size, max_size=max_size, mask_color=mask_color
        )
    
    def generate(self, segmentation_map=None):
        """
        Generate a mask using a mix of techniques.
        
        Args:
            segmentation_map (numpy.ndarray, optional): Segmentation map for object-aware masks
            
        Returns:
            numpy.ndarray: Binary mask
        """
        probs = []
        generators = []
        
        # Always include irregular masks
        probs.append(self.irregular_prob)
        generators.append(lambda: self.irregular_generator.generate())
        
        # Include object-aware masks if segmentation map is provided
        if segmentation_map is not None:
            if self.object_prob > 0:
                probs.append(self.object_prob)
                generators.append(lambda: self.object_generator.generate(segmentation_map))
            
            if self.boundary_prob > 0:
                probs.append(self.boundary_prob)
                generators.append(lambda: self.boundary_generator.generate(segmentation_map))
        
        # Always include damage simulation
        if self.damage_prob > 0:
            probs.append(self.damage_prob)
            generators.append(lambda: self.damage_generator.generate())
        
        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        # Choose a generator
        generator_idx = np.random.choice(len(generators), p=probs)
        generator = generators[generator_idx]
        
        return generator()

def get_mask_generator(config, height=512, width=512):
    """
    Factory function to create a mask generator based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        height (int): Height of masks
        width (int): Width of masks
        
    Returns:
        MaskGenerator: Configured mask generator
    """
    mask_type = config.get('type', 'mixed')
    mask_color = config.get('color', 1)
    min_size = config.get('min_size', 0.1)
    max_size = config.get('max_size', 0.5)
    
    if mask_type == 'irregular':
        return RandomIrregularMaskGenerator(
            height=height, width=width,
            min_size=min_size, max_size=max_size,
            mask_color=mask_color
        )
    elif mask_type == 'object':
        return ObjectAwareMaskGenerator(
            height=height, width=width,
            mask_color=mask_color
        )
    elif mask_type == 'boundary':
        return BoundaryCrossingMaskGenerator(
            height=height, width=width,
            min_size=min_size, max_size=max_size,
            mask_color=mask_color
        )
    elif mask_type == 'damage':
        damage_type = config.get('damage_type', 'mixed')
        return DamageSimulationMaskGenerator(
            height=height, width=width,
            damage_type=damage_type,
            mask_color=mask_color
        )
    else:  # mixed or default
        return MixedMaskGenerator(
            height=height, width=width,
            mask_color=mask_color,
            irregular_prob=config.get('irregular_prob', 0.6),
            boundary_prob=config.get('boundary_prob', 0.2),
            object_prob=config.get('object_prob', 0.3),
            damage_prob=config.get('damage_prob', 0.3),
            min_size=min_size, max_size=max_size
        ) 