#!/usr/bin/env python3
"""
Advanced rice seed counter with color-based trash filtering.
Separates filled seeds (lighter) from empty husks/trash (darker).
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from scipy import ndimage

def count_seeds_with_color_filter(image_path, 
                                  min_seed_size=150,
                                  max_seed_size=2500,
                                  brightness_threshold=None,
                                  separation_aggressiveness=0.4,
                                  show_debug=True):
    """
    Count filled rice seeds, filtering out darker trash/empty husks.
    
    Args:
        image_path: Path to image
        min_seed_size: Minimum seed area in pixels
        max_seed_size: Maximum seed area in pixels  
        brightness_threshold: Pixel value threshold (0-255). 
                            Below this = trash (darker), above = seeds (lighter)
                            If None, auto-calculated
        separation_aggressiveness: 0.1-0.9. Lower = more aggressive separation
        show_debug: Save debug images showing filtering steps
    """
    print(f"\nProcessing: {image_path}")
    print("="*70)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return None
    
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")
    
    # Convert to different color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Step 1: Find ALL objects (seeds + trash)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Adaptive threshold works better for varying lighting
    binary_adaptive = cv2.adaptiveThreshold(blurred, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 21, 8)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 2: Watershed to separate touching objects
    dist_transform = cv2.distanceTransform(binary_clean, cv2.DIST_L2, 5)
    
    # More aggressive watershed for dense seeds
    _, sure_fg = cv2.threshold(dist_transform, 
                              separation_aggressiveness * dist_transform.max(), 
                              255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Find markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    
    unknown = cv2.subtract(binary_clean, sure_fg)
    markers[unknown == 255] = 0
    
    # Apply watershed
    image_copy = image.copy()
    markers = cv2.watershed(image_copy, markers)
    
    # Step 3: Filter by SIZE and COLOR
    unique_labels = np.unique(markers)
    unique_labels = unique_labels[(unique_labels > 1) & (unique_labels != -1)]
    
    # Auto-calculate brightness threshold if not provided
    if brightness_threshold is None:
        # Sample the grayscale values of detected regions
        region_brightnesses = []
        for label in unique_labels[:50]:  # Sample first 50 regions
            mask = (markers == label)
            mean_brightness = gray[mask].mean()
            region_brightnesses.append(mean_brightness)
        
        if region_brightnesses:
            # Use Otsu's method on region brightnesses to separate light/dark
            hist, bins = np.histogram(region_brightnesses, bins=50)
            brightness_threshold = np.median(region_brightnesses) * 0.85  # Slightly below median
            print(f"Auto-calculated brightness threshold: {brightness_threshold:.1f}")
    
    filled_seeds = []
    trash_objects = []
    
    for label in unique_labels:
        # Create mask for this region
        region_mask = (markers == label).astype(np.uint8)
        area = np.sum(region_mask)
        
        # Filter by size first
        if not (min_seed_size <= area <= max_seed_size):
            continue
        
        # Calculate mean brightness of this region
        mean_brightness = gray[region_mask == 1].mean()
        
        # Separate based on brightness
        # Filled seeds are LIGHTER (higher brightness values)
        # Empty husks/trash are DARKER (lower brightness values)
        if mean_brightness >= brightness_threshold:
            filled_seeds.append((label, mean_brightness, area))
        else:
            trash_objects.append((label, mean_brightness, area))
    
    seed_count = len(filled_seeds)
    trash_count = len(trash_objects)
    total_detected = seed_count + trash_count
    
    # Create visualizations
    result_img = image.copy()
    trash_img = image.copy()
    combined_img = image.copy()
    
    # Draw filled seeds in GREEN
    for label, brightness, area in filled_seeds:
        mask = (markers == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
        cv2.drawContours(combined_img, contours, -1, (0, 255, 0), 2)
    
    # Draw trash in RED
    for label, brightness, area in trash_objects:
        mask = (markers == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(trash_img, contours, -1, (0, 0, 255), 2)
        cv2.drawContours(combined_img, contours, -1, (0, 0, 255), 2)
    
    # Add text overlays
    cv2.putText(result_img, f"FILLED SEEDS: {seed_count}", 
               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    
    cv2.putText(trash_img, f"TRASH/EMPTY: {trash_count}", 
               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    cv2.putText(combined_img, f"GREEN=Seeds({seed_count}) RED=Trash({trash_count})", 
               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(combined_img, f"Brightness threshold: {brightness_threshold:.1f}", 
               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save results
    output_dir = Path(image_path).parent
    output_name = Path(image_path).stem
    
    seeds_path = output_dir / f"{output_name}_SEEDS.png"
    cv2.imwrite(str(seeds_path), result_img)
    print(f"\n✓ Saved seeds only: {seeds_path.name}")
    
    trash_path = output_dir / f"{output_name}_TRASH.png"
    cv2.imwrite(str(trash_path), trash_img)
    print(f"✓ Saved trash only: {trash_path.name}")
    
    combined_path = output_dir / f"{output_name}_COMBINED.png"
    cv2.imwrite(str(combined_path), combined_img)
    print(f"✓ Saved combined: {combined_path.name}")
    
    if show_debug:
        # Save binary and distance transform
        debug_binary = output_dir / f"{output_name}_debug_binary.png"
        cv2.imwrite(str(debug_binary), binary_clean)
        
        # Create brightness visualization
        brightness_vis = np.zeros_like(image)
        for label, brightness, area in filled_seeds:
            mask = (markers == label)
            brightness_vis[mask] = (0, 255, 0)  # Green for seeds
        for label, brightness, area in trash_objects:
            mask = (markers == label)
            brightness_vis[mask] = (0, 0, 255)  # Red for trash
        
        brightness_path = output_dir / f"{output_name}_debug_brightness.png"
        cv2.imwrite(str(brightness_path), brightness_vis)
        
        print(f"✓ Saved debug images")
    
    # Print statistics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Filled seeds (lighter): {seed_count}")
    print(f"Trash/empty (darker):   {trash_count}")
    print(f"Total objects:          {total_detected}")
    print(f"\nYour manual count:      416 seeds")
    print(f"Detected seeds:         {seed_count}")
    print(f"Difference:             {abs(seed_count - 416)} ({abs(seed_count - 416) / 416 * 100:.1f}%)")
    
    accuracy = (1 - abs(seed_count - 416) / 416) * 100
    if accuracy >= 90:
        print(f"\n✓ EXCELLENT! {accuracy:.1f}% accuracy!")
    elif accuracy >= 80:
        print(f"\n✓ GOOD! {accuracy:.1f}% accuracy!")
    elif seed_count < 416 * 0.8:
        print(f"\n⚠️  Still under-counting ({accuracy:.1f}% accuracy)")
        print("Try:")
        print(f"   - Decrease separation_aggressiveness (currently {separation_aggressiveness})")
        print(f"   - Decrease min_seed_size (currently {min_seed_size})")
        print(f"   - Increase brightness_threshold (currently {brightness_threshold:.1f})")
    else:
        print(f"\n⚠️  Over-counting ({accuracy:.1f}% accuracy)")
        print("Try:")
        print(f"   - Increase separation_aggressiveness (currently {separation_aggressiveness})")
        print(f"   - Increase min_seed_size (currently {min_seed_size})")
        print(f"   - Decrease brightness_threshold (currently {brightness_threshold:.1f})")
    
    print("="*70)
    
    return {
        'seed_count': seed_count,
        'trash_count': trash_count,
        'total': total_detected,
        'filled_seeds': filled_seeds,
        'trash_objects': trash_objects,
        'brightness_threshold': brightness_threshold
    }


def interactive_tune(image_path):
    """Interactive parameter tuning."""
    print("\n" + "="*70)
    print("INTERACTIVE PARAMETER TUNING")
    print("="*70)
    print("Testing different parameter combinations...\n")
    
    best_diff = float('inf')
    best_params = None
    best_count = 0
    
    # Test combinations
    for sep_agg in [0.2, 0.3, 0.4, 0.5]:
        for min_size in [100, 150, 200]:
            for brightness in [None, 90, 100, 110, 120]:
                print(f"\nTesting: sep={sep_agg}, min={min_size}, brightness={brightness}")
                
                result = count_seeds_with_color_filter(
                    image_path, 
                    min_seed_size=min_size,
                    max_seed_size=2500,
                    brightness_threshold=brightness,
                    separation_aggressiveness=sep_agg,
                    show_debug=False
                )
                
                if result is None:
                    continue
                
                count = result['seed_count']
                diff = abs(count - 416)
                accuracy = (1 - diff / 416) * 100
                
                print(f"  → Seeds: {count}, Trash: {result['trash_count']}, "
                      f"Diff: {diff}, Accuracy: {accuracy:.1f}%")
                
                if diff < best_diff:
                    best_diff = diff
                    best_params = (sep_agg, min_size, brightness)
                    best_count = count
    
    print("\n" + "="*70)
    print("BEST PARAMETERS FOUND:")
    print(f"  separation_aggressiveness: {best_params[0]}")
    print(f"  min_seed_size: {best_params[1]}")
    print(f"  brightness_threshold: {best_params[2]}")
    print(f"  Result: {best_count} seeds (diff: {best_diff}, accuracy: {(1-best_diff/416)*100:.1f}%)")
    print("="*70)
    
    return best_params


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python color_filter_counter.py <image_path>                    # Default params")
        print("  python color_filter_counter.py <image_path> --tune             # Auto-tune")
        print("  python color_filter_counter.py <image_path> --sep 0.3 --min 150 --brightness 110")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if '--tune' in sys.argv:
        # Interactive tuning
        best_params = interactive_tune(image_path)
        print(f"\nRun with best parameters:")
        print(f"python color_filter_counter.py '{image_path}' --sep {best_params[0]} "
              f"--min {best_params[1]} --brightness {best_params[2]}")
    else:
        # Parse parameters
        sep_agg = 0.3  # Default: aggressive separation
        min_size = 150
        max_size = 2500
        brightness = None  # Auto-calculate
        
        if '--sep' in sys.argv:
            idx = sys.argv.index('--sep')
            sep_agg = float(sys.argv[idx + 1])
        
        if '--min' in sys.argv:
            idx = sys.argv.index('--min')
            min_size = int(sys.argv[idx + 1])
        
        if '--max' in sys.argv:
            idx = sys.argv.index('--max')
            max_size = int(sys.argv[idx + 1])
        
        if '--brightness' in sys.argv:
            idx = sys.argv.index('--brightness')
            brightness = float(sys.argv[idx + 1])
        
        # Count seeds
        result = count_seeds_with_color_filter(
            image_path,
            min_seed_size=min_size,
            max_seed_size=max_size,
            brightness_threshold=brightness,
            separation_aggressiveness=sep_agg,
            show_debug=True
        )
        
        if result:
            print(f"\n✓ Check the results in: {Path(image_path).parent}")
            print(f"\n💡 Look at:")
            print(f"   - {Path(image_path).stem}_SEEDS.png (green = filled seeds)")
            print(f"   - {Path(image_path).stem}_TRASH.png (red = trash/empty)")
            print(f"   - {Path(image_path).stem}_COMBINED.png (both overlaid)")
