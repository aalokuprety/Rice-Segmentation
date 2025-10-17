#!/usr/bin/env python3
"""
Improved rice seed counter with watershed separation.
Handles densely packed/touching seeds.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from scipy import ndimage
from skimage import measure, morphology

def count_dense_seeds(image_path, min_seed_size=200, max_seed_size=2000, 
                     separation_distance=8, show_debug=True):
    """
    Count seeds in a densely packed image using watershed separation.
    
    Args:
        image_path: Path to image
        min_seed_size: Minimum seed area in pixels (adjust based on your seeds)
        max_seed_size: Maximum seed area in pixels
        separation_distance: Distance for watershed (lower = more aggressive separation)
        show_debug: Save debug images
    """
    print(f"\nProcessing: {image_path}")
    print("="*70)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return None
    
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Blur slightly
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Threshold - try both Otsu and Adaptive
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 5)
    
    # Combine both for better results
    binary = cv2.bitwise_and(binary_otsu, binary_adaptive)
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Find sure foreground (seed centers)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Find markers for watershed
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all markers so background is not 0, but 1
    markers = markers + 1
    
    # Mark unknown region as 0
    unknown = cv2.subtract(binary, sure_fg)
    markers[unknown == 255] = 0
    
    # Apply watershed
    image_copy = image.copy()
    markers = cv2.watershed(image_copy, markers)
    
    # Count unique labels (excluding boundary -1 and background 1)
    unique_labels = np.unique(markers)
    unique_labels = unique_labels[(unique_labels > 1) & (unique_labels != -1)]
    
    # Filter by size
    valid_seeds = []
    for label in unique_labels:
        region_mask = (markers == label).astype(np.uint8)
        area = np.sum(region_mask)
        
        if min_seed_size <= area <= max_seed_size:
            valid_seeds.append(label)
    
    seed_count = len(valid_seeds)
    
    # Create visualization
    result_img = image.copy()
    
    # Color each seed differently
    for i, label in enumerate(valid_seeds):
        # Create mask for this seed
        mask = (markers == label).astype(np.uint8)
        
        # Find contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contour in green
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
        
        # Add number at center
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Only show number for every 10th seed to avoid clutter
            if i % 10 == 0:
                cv2.putText(result_img, str(i+1), (cx-10, cy+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Add text overlay
    cv2.putText(result_img, f"DETECTED: {seed_count} seeds", 
               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.putText(result_img, f"(min size: {min_seed_size}, max: {max_seed_size})", 
               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save results
    output_dir = Path(image_path).parent
    output_name = Path(image_path).stem
    
    # Save main result
    result_path = output_dir / f"{output_name}_counted.png"
    cv2.imwrite(str(result_path), result_img)
    print(f"\n✓ Saved result: {result_path.name}")
    
    if show_debug:
        # Save debug images
        debug_binary = output_dir / f"{output_name}_debug_binary.png"
        cv2.imwrite(str(debug_binary), binary)
        
        debug_dist = output_dir / f"{output_name}_debug_distance.png"
        dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(debug_dist), dist_vis)
        
        debug_markers = output_dir / f"{output_name}_debug_markers.png"
        markers_vis = cv2.normalize(markers.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        markers_color = cv2.applyColorMap(markers_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(debug_markers), markers_color)
        
        print(f"✓ Saved debug images: binary, distance, markers")
    
    # Print statistics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Seeds detected: {seed_count}")
    print(f"Your count: 416")
    print(f"Difference: {abs(seed_count - 416)} ({abs(seed_count - 416) / 416 * 100:.1f}%)")
    
    if seed_count < 416 * 0.8:
        print("\n⚠️  Under-counting! Try:")
        print(f"   - Decrease min_seed_size (currently {min_seed_size})")
        print(f"   - Decrease separation_distance (currently {separation_distance})")
    elif seed_count > 416 * 1.2:
        print("\n⚠️  Over-counting! Try:")
        print(f"   - Increase min_seed_size (currently {min_seed_size})")
        print(f"   - Increase max_seed_size if trash is larger")
    else:
        print("\n✓ Good accuracy! (within 20% of true count)")
    
    print("="*70)
    
    return {
        'seed_count': seed_count,
        'markers': markers,
        'binary': binary,
        'result_image': result_img
    }


def auto_tune_parameters(image_path, true_count=416):
    """Try different parameter combinations to find best match."""
    print("\n" + "="*70)
    print("AUTO-TUNING PARAMETERS")
    print("="*70)
    print(f"Target count: {true_count} seeds\n")
    
    best_diff = float('inf')
    best_params = None
    best_count = 0
    
    # Test different min_seed_size values
    for min_size in [100, 150, 200, 250, 300, 400]:
        for max_size in [1500, 2000, 2500, 3000]:
            if max_size <= min_size * 2:
                continue
            
            result = count_dense_seeds(image_path, min_size, max_size, 
                                      separation_distance=8, show_debug=False)
            
            if result is None:
                continue
            
            count = result['seed_count']
            diff = abs(count - true_count)
            
            print(f"min={min_size:3d}, max={max_size:4d} → {count:3d} seeds (diff: {diff:3d}, {diff/true_count*100:5.1f}%)")
            
            if diff < best_diff:
                best_diff = diff
                best_params = (min_size, max_size)
                best_count = count
    
    print("\n" + "="*70)
    print(f"BEST PARAMETERS:")
    print(f"  min_seed_size: {best_params[0]}")
    print(f"  max_seed_size: {best_params[1]}")
    print(f"  Result: {best_count} seeds (diff: {best_diff}, {best_diff/true_count*100:.1f}%)")
    print("="*70)
    
    return best_params


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python seed_counter.py <image_path>                    # Count with default params")
        print("  python seed_counter.py <image_path> --auto-tune        # Find best parameters")
        print("  python seed_counter.py <image_path> --min 200 --max 2000  # Custom params")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if '--auto-tune' in sys.argv:
        # Auto-tune to find best parameters
        best_params = auto_tune_parameters(image_path, true_count=416)
        print(f"\nNow run with best parameters:")
        print(f"python seed_counter.py '{image_path}' --min {best_params[0]} --max {best_params[1]}")
    else:
        # Parse custom parameters
        min_seed = 200
        max_seed = 2000
        
        if '--min' in sys.argv:
            idx = sys.argv.index('--min')
            min_seed = int(sys.argv[idx + 1])
        
        if '--max' in sys.argv:
            idx = sys.argv.index('--max')
            max_seed = int(sys.argv[idx + 1])
        
        # Count seeds
        result = count_dense_seeds(image_path, min_seed, max_seed, 
                                   separation_distance=8, show_debug=True)
        
        if result:
            print(f"\n✓ Open the folder to see results!")
            print(f"  {Path(image_path).parent}")
