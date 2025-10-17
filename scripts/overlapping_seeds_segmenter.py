#!/usr/bin/env python3
"""
Overlapping Seeds Segmentation

Specialized methods for separating touching/overlapping rice seeds.
Uses advanced watershed and morphological operations.
"""

import cv2
import numpy as np
from skimage import measure, morphology
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
import matplotlib.pyplot as plt


class OverlappingSeedsSegmenter:
    """
    Specialized segmenter for handling overlapping/touching rice seeds.
    Uses aggressive watershed and morphological separation.
    """
    
    def __init__(self, min_seed_area: int = 300, max_seed_area: int = 8000,
                 min_distance: int = 15, sensitivity: float = 0.5):
        """
        Initialize the overlapping seeds segmenter.
        
        Args:
            min_seed_area: Minimum area for valid seed (pixels)
            max_seed_area: Maximum area for valid seed (pixels)
            min_distance: Minimum distance between seed centers (lower = more sensitive)
            sensitivity: Watershed sensitivity (0.1-1.0, lower = more aggressive separation)
        """
        self.min_seed_area = min_seed_area
        self.max_seed_area = max_seed_area
        self.min_distance = min_distance
        self.sensitivity = sensitivity
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better separation of touching seeds.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def create_binary_mask(self, gray_image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """
        Create binary mask with optimal settings for touching seeds.
        
        Args:
            gray_image: Grayscale image
            method: 'otsu', 'adaptive', or 'combined'
            
        Returns:
            Binary mask
        """
        if method == 'otsu':
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 35, 10
            )
        
        elif method == 'combined':
            # Use both methods and combine
            _, otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            adaptive = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 35, 10
            )
            binary = cv2.bitwise_and(otsu, adaptive)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return binary
    
    def separate_touching_seeds(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply advanced watershed algorithm to separate touching seeds.
        
        Args:
            binary_mask: Binary mask with seeds
            
        Returns:
            Labeled image with separated seeds
        """
        # Distance transform - measures distance to nearest background pixel
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # Normalize distance transform
        dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Find local maxima (seed centers)
        # Lower min_distance = more sensitive to separate touching seeds
        local_maxima = peak_local_maxima(
            dist_norm,
            min_distance=self.min_distance,
            threshold_abs=self.sensitivity * dist_norm.max(),
            exclude_border=False
        )
        
        # Create markers for watershed
        markers = np.zeros(dist_transform.shape, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Dilate markers slightly to ensure they're in the center of seeds
        markers = ndimage.maximum_filter(markers, size=3)
        
        # Apply watershed algorithm
        # Watershed treats the distance transform as a topographic surface
        # It "floods" from the markers, creating boundaries where regions meet
        labels = watershed(-dist_transform, markers, mask=binary_mask)
        
        return labels
    
    def filter_seeds(self, labeled_image: np.ndarray) -> np.ndarray:
        """
        Filter out invalid seeds based on area and shape.
        
        Args:
            labeled_image: Labeled image with separated seeds
            
        Returns:
            Filtered labeled image
        """
        # Get properties of each region
        props = measure.regionprops(labeled_image)
        
        # Create new labels for valid seeds only
        filtered_labels = np.zeros_like(labeled_image)
        new_label = 1
        
        for prop in props:
            area = prop.area
            
            # Filter by area
            if area < self.min_seed_area or area > self.max_seed_area:
                continue
            
            # Optional: Filter by aspect ratio (for rice seeds: elongated)
            # Uncomment if needed
            # min_axis = prop.minor_axis_length
            # maj_axis = prop.major_axis_length
            # if maj_axis > 0:
            #     aspect_ratio = maj_axis / (min_axis + 1e-6)
            #     if aspect_ratio < 1.2 or aspect_ratio > 8.0:
            #         continue
            
            # Keep this seed
            filtered_labels[labeled_image == prop.label] = new_label
            new_label += 1
        
        return filtered_labels
    
    def segment_overlapping_seeds(self, image: np.ndarray, 
                                  threshold_method: str = 'adaptive',
                                  visualize: bool = False) -> dict:
        """
        Complete pipeline to segment overlapping seeds.
        
        Args:
            image: Input BGR image
            threshold_method: 'otsu', 'adaptive', or 'combined'
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with results
        """
        # Preprocess
        gray = self.preprocess_image(image)
        
        # Create binary mask
        binary = self.create_binary_mask(gray, method=threshold_method)
        
        # Separate touching seeds
        labels = self.separate_touching_seeds(binary)
        
        # Filter invalid seeds
        labels_filtered = self.filter_seeds(labels)
        
        # Count seeds
        seed_count = labels_filtered.max()
        
        # Get properties
        props = measure.regionprops(labels_filtered)
        
        # Create visualization if requested
        vis_image = None
        if visualize:
            vis_image = self.create_visualization(image, labels_filtered, props)
        
        results = {
            'seed_count': seed_count,
            'binary_mask': binary,
            'labeled_mask': labels_filtered,
            'labels_raw': labels,
            'properties': props,
            'visualization': vis_image
        }
        
        return results
    
    def create_visualization(self, original_image: np.ndarray, 
                           labeled_image: np.ndarray,
                           props: list) -> np.ndarray:
        """
        Create visualization showing separated seeds.
        
        Args:
            original_image: Original input image
            labeled_image: Labeled image with separated seeds
            props: Region properties
            
        Returns:
            Visualization image
        """
        # Create RGB overlay
        from skimage import color as skcolor
        
        # Convert original to RGB if needed
        if len(original_image.shape) == 2:
            vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 4:
            vis = cv2.cvtColor(original_image, cv2.COLOR_BGRA2RGB)
        else:
            vis = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Create colored label overlay
        colored_labels = skcolor.label2rgb(labeled_image, vis, alpha=0.4, bg_label=0)
        vis_overlay = (colored_labels * 255).astype(np.uint8)
        
        # Draw contours and numbers
        for i, prop in enumerate(props):
            # Get contour
            y, x = prop.coords.T
            
            # Draw boundary
            contour_mask = np.zeros(labeled_image.shape, dtype=np.uint8)
            contour_mask[labeled_image == prop.label] = 255
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_overlay, contours, -1, (0, 255, 0), 2)
            
            # Add number at centroid
            cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
            cv2.putText(vis_overlay, str(i+1), (cx-10, cy+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_overlay
    
    def tune_parameters(self, image: np.ndarray, min_distance_range: tuple = (10, 30),
                       sensitivity_range: tuple = (0.3, 0.7)) -> dict:
        """
        Automatically tune parameters for best separation.
        
        Args:
            image: Sample image
            min_distance_range: Range to test for min_distance
            sensitivity_range: Range to test for sensitivity
            
        Returns:
            Best parameters
        """
        best_params = {}
        best_score = 0
        
        gray = self.preprocess_image(image)
        binary = self.create_binary_mask(gray, method='adaptive')
        
        # Test different parameter combinations
        for min_dist in range(min_distance_range[0], min_distance_range[1], 5):
            for sens in np.arange(sensitivity_range[0], sensitivity_range[1], 0.1):
                self.min_distance = min_dist
                self.sensitivity = sens
                
                labels = self.separate_touching_seeds(binary)
                labels_filtered = self.filter_seeds(labels)
                
                seed_count = labels_filtered.max()
                
                # Score: prefer reasonable seed counts, penalize extreme values
                if 20 < seed_count < 200:
                    score = seed_count
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'min_distance': min_dist,
                            'sensitivity': sens,
                            'seed_count': seed_count
                        }
        
        return best_params


def demo_overlapping_segmentation(image_path: str):
    """
    Demonstrate overlapping seed segmentation.
    
    Args:
        image_path: Path to test image
    """
    # Load image
    image = cv2.imread(image_path)
    
    # Create segmenter
    segmenter = OverlappingSeedsSegmenter(
        min_seed_area=300,
        max_seed_area=8000,
        min_distance=15,
        sensitivity=0.5
    )
    
    # Segment
    results = segmenter.segment_overlapping_seeds(image, visualize=True)
    
    # Display results
    print(f"Detected {results['seed_count']} seeds")
    
    # Show visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Binary mask
    axes[0, 1].imshow(results['binary_mask'], cmap='gray')
    axes[0, 1].set_title('Binary Mask')
    axes[0, 1].axis('off')
    
    # Raw labels (before filtering)
    axes[1, 0].imshow(results['labels_raw'], cmap='nipy_spectral')
    axes[1, 0].set_title(f'Raw Separation ({results["labels_raw"].max()} regions)')
    axes[1, 0].axis('off')
    
    # Final result
    axes[1, 1].imshow(results['visualization'])
    axes[1, 1].set_title(f'Final Result ({results["seed_count"]} seeds)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Overlapping Seeds Segmentation Module")
    print("="*50)
    print("\nUsage:")
    print("  from overlapping_seeds_segmenter import OverlappingSeedsSegmenter")
    print("  segmenter = OverlappingSeedsSegmenter()")
    print("  results = segmenter.segment_overlapping_seeds(image)")
    print("  print(f'Found {results[\"seed_count\"]} seeds')")
