#!/usr/bin/env python3
"""
Rice Seeds Segmentation Methods

This module contains various computer vision approaches for segmenting rice seeds from images.
Includes traditional methods like thresholding, edge detection, watershed, and region growing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, filters, segmentation
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    from scipy.ndimage import maximum_filter
    from scipy.ndimage import generate_binary_structure
    def peak_local_maxima(image, min_distance=1, threshold_abs=0, exclude_border=True):
        """Fallback implementation for peak_local_maxima."""
        struct = generate_binary_structure(2, 2)
        local_max = maximum_filter(image, footprint=struct) == image
        background = (image == 0)
        eroded_background = morphology.binary_erosion(background, structure=struct, border_value=1)
        detected_peaks = local_max ^ eroded_background
        if threshold_abs > 0:
            detected_peaks = detected_peaks & (image > threshold_abs)
        peaks = np.column_stack(np.where(detected_peaks))
        return peaks
from skimage.segmentation import watershed
from scipy import ndimage
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union


class RiceSeedSegmenter:
    """
    A comprehensive class for rice seed segmentation using various computer vision techniques.
    """
    
    def __init__(self, min_seed_area: int = 50, max_seed_area: int = 5000, 
                 min_aspect_ratio: float = 1.5, max_aspect_ratio: float = 6.0):
        """
        Initialize the rice seed segmenter with default parameters.
        
        Args:
            min_seed_area: Minimum area for a valid seed (pixels)
            max_seed_area: Maximum area for a valid seed (pixels)
            min_aspect_ratio: Minimum aspect ratio for rice seeds
            max_aspect_ratio: Maximum aspect ratio for rice seeds
        """
        self.min_seed_area = min_seed_area
        self.max_seed_area = max_seed_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
    
    def load_and_preprocess(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load an image and apply preprocessing steps.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (original_image, preprocessed_image)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Apply median filter to further reduce noise
        preprocessed = cv2.medianBlur(enhanced, 3)
        
        return original, preprocessed
    
    def otsu_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Otsu's thresholding method.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Binary image
        """
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def adaptive_segmentation(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Apply adaptive thresholding.
        
        Args:
            image: Preprocessed grayscale image
            method: 'mean' or 'gaussian'
            
        Returns:
            Binary image
        """
        if method == 'gaussian':
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        else:
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        return binary
    
    def canny_edge_segmentation(self, image: np.ndarray, low_threshold: int = 50, 
                               high_threshold: int = 150) -> np.ndarray:
        """
        Apply Canny edge detection followed by morphological operations.
        
        Args:
            image: Preprocessed grayscale image
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            
        Returns:
            Binary image with filled regions
        """
        # Apply Canny edge detection
        edges = cv2.Canny(image, low_threshold, high_threshold)
        
        # Close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Fill holes to create solid regions
        filled = ndimage.binary_fill_holes(closed_edges).astype(np.uint8) * 255
        
        return filled
    
    def morphological_cleanup(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up binary image.
        
        Args:
            binary_image: Binary input image
            
        Returns:
            Cleaned binary image
        """
        # Define kernels
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Opening to remove noise
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_medium)
        
        # Closing to fill holes
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)
        
        return cleaned
    
    def watershed_segmentation(self, binary_image: np.ndarray, min_distance: int = 20) -> np.ndarray:
        """
        Apply watershed algorithm to separate touching seeds.
        
        Args:
            binary_image: Binary input image
            min_distance: Minimum distance between seed centers
            
        Returns:
            Labeled image with individual seeds
        """
        # Compute distance transform
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        
        # Find local maxima (seed centers)
        local_maxima = peak_local_maxima(dist_transform, min_distance=min_distance, 
                                       threshold_abs=0.3)
        
        # Create markers for watershed
        markers = np.zeros(dist_transform.shape, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Apply watershed
        labels = watershed(-dist_transform, markers, mask=binary_image)
        
        return labels
    
    def connected_components_segmentation(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Use connected components analysis for segmentation.
        
        Args:
            binary_image: Binary input image
            
        Returns:
            Labeled image with individual components
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8)
        
        # Filter out background and small components
        filtered_labels = np.zeros_like(labels)
        new_label = 1
        
        for label in range(1, num_labels):  # Skip background (0)
            if stats[label, cv2.CC_STAT_AREA] >= self.min_seed_area:
                filtered_labels[labels == label] = new_label
                new_label += 1
        
        return filtered_labels
    
    def region_growing_segmentation(self, image: np.ndarray, seeds: Optional[List[Tuple[int, int]]] = None,
                                  threshold: float = 10) -> np.ndarray:
        """
        Apply region growing algorithm.
        
        Args:
            image: Grayscale input image
            seeds: List of seed points (y, x). If None, automatically detected
            threshold: Threshold for region growing
            
        Returns:
            Labeled image with grown regions
        """
        h, w = image.shape
        labels = np.zeros((h, w), dtype=np.int32)
        
        if seeds is None:
            # Automatically detect seeds using local minima
            binary = self.otsu_segmentation(image)
            cleaned = self.morphological_cleanup(binary)
            dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
            seeds = peak_local_maxima(dist_transform, min_distance=20, threshold_abs=0.3)
        
        label_id = 1
        
        for seed_y, seed_x in seeds:
            if labels[seed_y, seed_x] != 0:  # Already labeled
                continue
            
            # Initialize region growing
            seed_value = float(image[seed_y, seed_x])
            stack = [(seed_y, seed_x)]
            region_pixels = []
            
            while stack:
                y, x = stack.pop()
                
                if (y < 0 or y >= h or x < 0 or x >= w or 
                    labels[y, x] != 0):  # Out of bounds or already labeled
                    continue
                
                pixel_value = float(image[y, x])
                if abs(pixel_value - seed_value) <= threshold:
                    labels[y, x] = label_id
                    region_pixels.append((y, x))
                    
                    # Add neighbors to stack
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy != 0 or dx != 0:  # Skip center pixel
                                stack.append((y + dy, x + dx))
            
            # Only keep region if it's large enough
            if len(region_pixels) >= self.min_seed_area:
                label_id += 1
            else:
                # Remove small region
                for y, x in region_pixels:
                    labels[y, x] = 0
        
        return labels
    
    def filter_seeds_by_geometry(self, labels: np.ndarray) -> np.ndarray:
        """
        Filter detected seeds based on geometric properties.
        
        Args:
            labels: Labeled image
            
        Returns:
            Filtered labeled image
        """
        props = measure.regionprops(labels)
        filtered_labels = np.zeros_like(labels)
        new_label = 1
        
        for prop in props:
            area = prop.area
            
            # Check area constraints
            if area < self.min_seed_area or area > self.max_seed_area:
                continue
            
            # Check aspect ratio (rice seeds are elongated)
            if prop.minor_axis_length > 0:
                aspect_ratio = prop.major_axis_length / prop.minor_axis_length
                if (aspect_ratio < self.min_aspect_ratio or 
                    aspect_ratio > self.max_aspect_ratio):
                    continue
            
            # Check solidity (seeds should be fairly solid)
            if prop.solidity < 0.6:
                continue
            
            # Keep this seed
            filtered_labels[labels == prop.label] = new_label
            new_label += 1
        
        return filtered_labels
    
    def extract_features(self, labels: np.ndarray, 
                        original_image: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Extract morphological features from segmented seeds.
        
        Args:
            labels: Labeled segmentation image
            original_image: Original image for intensity features
            
        Returns:
            DataFrame with extracted features
        """
        props = measure.regionprops(labels, intensity_image=original_image)
        features = []
        
        for prop in props:
            feature_dict = {
                'label': prop.label,
                'area': prop.area,
                'perimeter': prop.perimeter,
                'centroid_y': prop.centroid[0],
                'centroid_x': prop.centroid[1],
                'major_axis_length': prop.major_axis_length,
                'minor_axis_length': prop.minor_axis_length,
                'eccentricity': prop.eccentricity,
                'solidity': prop.solidity,
                'extent': prop.extent,
                'orientation': prop.orientation,
                'equivalent_diameter': prop.equivalent_diameter
            }
            
            # Calculate additional features
            if prop.perimeter > 0:
                feature_dict['circularity'] = 4 * np.pi * prop.area / (prop.perimeter ** 2)
            else:
                feature_dict['circularity'] = 0
            
            if prop.minor_axis_length > 0:
                feature_dict['aspect_ratio'] = prop.major_axis_length / prop.minor_axis_length
            else:
                feature_dict['aspect_ratio'] = 0
            
            # Add intensity features if available
            if original_image is not None:
                feature_dict['mean_intensity'] = prop.mean_intensity
                feature_dict['min_intensity'] = prop.min_intensity
                feature_dict['max_intensity'] = prop.max_intensity
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def segment_image(self, image_path: Union[str, Path], method: str = 'watershed',
                     save_results: bool = True, output_dir: Optional[Path] = None) -> Dict:
        """
        Complete segmentation pipeline for a single image.
        
        Args:
            image_path: Path to input image
            method: Segmentation method ('otsu', 'adaptive', 'watershed', 'connected_components', 'region_growing', 'canny')
            save_results: Whether to save intermediate results
            output_dir: Directory to save results
            
        Returns:
            Dictionary with segmentation results
        """
        # Load and preprocess
        original, preprocessed = self.load_and_preprocess(image_path)
        
        # Apply initial segmentation
        if method == 'otsu':
            binary = self.otsu_segmentation(preprocessed)
            labels = self.connected_components_segmentation(binary)
        elif method == 'adaptive':
            binary = self.adaptive_segmentation(preprocessed)
            labels = self.connected_components_segmentation(binary)
        elif method == 'watershed':
            binary = self.otsu_segmentation(preprocessed)
            cleaned_binary = self.morphological_cleanup(binary)
            labels = self.watershed_segmentation(cleaned_binary)
        elif method == 'connected_components':
            binary = self.otsu_segmentation(preprocessed)
            cleaned_binary = self.morphological_cleanup(binary)
            labels = self.connected_components_segmentation(cleaned_binary)
        elif method == 'region_growing':
            labels = self.region_growing_segmentation(preprocessed)
            binary = (labels > 0).astype(np.uint8) * 255
        elif method == 'canny':
            binary = self.canny_edge_segmentation(preprocessed)
            labels = self.connected_components_segmentation(binary)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Filter seeds by geometry
        filtered_labels = self.filter_seeds_by_geometry(labels)
        
        # Extract features
        features_df = self.extract_features(filtered_labels, preprocessed)
        
        # Prepare results
        results = {
            'original_image': original,
            'preprocessed_image': preprocessed,
            'binary_image': binary if 'binary' in locals() else None,
            'labels': filtered_labels,
            'features': features_df,
            'seed_count': len(features_df),
            'method_used': method
        }
        
        # Save results if requested
        if save_results and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save images
            if results['binary_image'] is not None:
                cv2.imwrite(str(output_dir / 'binary.png'), results['binary_image'])
            
            colored_labels = measure.label2rgb(filtered_labels, bg_label=0)
            plt.imsave(str(output_dir / 'segmented.png'), colored_labels)
            
            # Save features
            if not features_df.empty:
                features_df.to_csv(str(output_dir / 'features.csv'), index=False)
        
        return results


# Example usage and comparison functions
def compare_segmentation_methods(image_path: Union[str, Path], 
                               methods: List[str] = None) -> Dict:
    """
    Compare different segmentation methods on the same image.
    
    Args:
        image_path: Path to input image
        methods: List of methods to compare
        
    Returns:
        Dictionary with results for each method
    """
    if methods is None:
        methods = ['otsu', 'adaptive', 'watershed', 'connected_components', 'canny']
    
    segmenter = RiceSeedSegmenter()
    results = {}
    
    for method in methods:
        try:
            result = segmenter.segment_image(image_path, method=method, save_results=False)
            results[method] = result
            print(f"{method}: {result['seed_count']} seeds detected")
        except Exception as e:
            print(f"Error with {method}: {str(e)}")
            results[method] = None
    
    return results


def visualize_comparison(results: Dict, figsize: Tuple[int, int] = (20, 12)):
    """
    Visualize comparison of different segmentation methods.
    
    Args:
        results: Results dictionary from compare_segmentation_methods
        figsize: Figure size for visualization
    """
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_methods = len(valid_results)
    
    if n_methods == 0:
        print("No valid results to visualize")
        return
    
    fig, axes = plt.subplots(2, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    for i, (method, result) in enumerate(valid_results.items()):
        # Show original in first row
        axes[0, i].imshow(result['original_image'])
        axes[0, i].set_title(f'{method.title()}: {result["seed_count"]} seeds')
        axes[0, i].axis('off')
        
        # Show segmentation in second row
        colored_labels = measure.label2rgb(result['labels'], bg_label=0)
        axes[1, i].imshow(colored_labels)
        axes[1, i].set_title(f'Segmentation Result')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Rice Seeds Segmentation Methods")
    print("=" * 40)
    print("Available methods:")
    print("- otsu: Otsu's thresholding + connected components")
    print("- adaptive: Adaptive thresholding + connected components")
    print("- watershed: Watershed algorithm")
    print("- connected_components: Connected components analysis")
    print("- region_growing: Region growing algorithm")
    print("- canny: Canny edge detection + filling")
    print("\nTo use this module:")
    print("1. Import the RiceSeedSegmenter class")
    print("2. Create an instance: segmenter = RiceSeedSegmenter()")
    print("3. Segment an image: results = segmenter.segment_image('path/to/image.jpg')")