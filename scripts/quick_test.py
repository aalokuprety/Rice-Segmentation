#!/usr/bin/env python3
"""Quick test script for rice seed segmentation"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rice_segmentation_methods import RiceSeedSegmenter

def quick_test(image_path):
    """Test segmentation on an image."""
    print(f"\nTesting segmentation on: {image_path}")
    print("="*60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image")
        return
    
    print(f"Image loaded: {image.shape[0]}x{image.shape[1]} pixels")
    
    # Create segmenter
    segmenter = RiceSeedSegmenter(
        min_seed_area=300,
        max_seed_area=8000
    )
    
    # Preprocess image
    print("Preprocessing...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed = clahe.apply(blurred)
    preprocessed = cv2.medianBlur(preprocessed, 3)
    
    # Try different methods
    print("\nTesting different segmentation methods...")
    print("-"*60)
    
    methods = ['otsu', 'adaptive', 'watershed']
    results = {}
    
    for method in methods:
        try:
            print(f"\n{method.upper()} method:")
            
            # Get binary mask using different methods
            if method == 'otsu':
                binary = segmenter.otsu_segmentation(preprocessed)
                binary_inv = cv2.bitwise_not(binary)  # Invert if needed
            elif method == 'adaptive':
                binary = segmenter.adaptive_segmentation(preprocessed)
                binary_inv = cv2.bitwise_not(binary)
            elif method == 'watershed':
                binary = segmenter.otsu_segmentation(preprocessed)  # Start with otsu
                binary_inv = cv2.bitwise_not(binary)
                labels = segmenter.apply_watershed(binary_inv, min_distance=15)
            else:
                continue
            
            # For watershed, use labels directly; for others, find connected components
            if method == 'watershed':
                num_seeds = labels.max()
                final_mask = (labels > 0).astype(np.uint8) * 255
            else:
                # Find connected components
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
                # Filter by area
                valid_labels = []
                for i in range(1, num_labels):
                    if segmenter.min_seed_area <= stats[i, cv2.CC_STAT_AREA] <= segmenter.max_seed_area:
                        valid_labels.append(i)
                num_seeds = len(valid_labels)
                final_mask = binary_inv
            
            results[method] = {'seed_count': num_seeds, 'binary_mask': final_mask}
            
            print(f"  ✓ Detected {num_seeds} seeds")
            
            # Save visualization
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_{method}.png"
            
            # Create simple overlay
            overlay = image.copy()
            mask_colored = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:,:,1] = 0  # Remove green, keep red and blue
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
            # Add text
            cv2.putText(overlay, f"{method.upper()}: {num_seeds} seeds", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            cv2.imwrite(str(output_path), overlay)
            print(f"  Saved: {output_path.name}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for method, result in results.items():
        print(f"{method.upper():15} : {result['seed_count']:3d} seeds")
    
    print("\n✓ Check the output images in the same folder as your test image!")
    print(f"  Location: {Path(image_path).parent}")
    
    # Recommend best method
    if results:
        best_method = max(results.items(), key=lambda x: x[1]['seed_count'])
        print(f"\n💡 Recommendation: Use '{best_method[0]}' method")
        print(f"   (Detected {best_method[1]['seed_count']} seeds)")
    
    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <image_path>")
        print("Example: python quick_test.py 'data/raw/variety_A/test image.jpeg'")
        sys.exit(1)
    
    image_path = sys.argv[1]
    quick_test(image_path)
