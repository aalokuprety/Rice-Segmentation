#!/usr/bin/env python3
"""
Genotype Screening with Watershed Segmentation

Fast processing for 100+ genotypes with overlapping/touching seeds.
Uses advanced watershed algorithm for automatic seed separation.
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from genotype_screening import GenotypeScreeningProcessor
from overlapping_seeds_segmenter import OverlappingSeedsSegmenter
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm


class WatershedGenotypeProcessor(GenotypeScreeningProcessor):
    """Extended genotype processor using watershed for overlapping seeds."""
    
    def __init__(self, output_dir: str = '../results/genotype_screening',
                 min_distance: int = 15, sensitivity: float = 0.5):
        """
        Initialize watershed-based genotype processor.
        
        Args:
            output_dir: Output directory
            min_distance: Minimum distance between seed centers (10-30)
            sensitivity: Watershed sensitivity (0.3-0.7, lower = more aggressive)
        """
        super().__init__(output_dir)
        self.watershed_segmenter = OverlappingSeedsSegmenter(
            min_seed_area=300,
            max_seed_area=8000,
            min_distance=min_distance,
            sensitivity=sensitivity
        )
    
    def process_single_image(self, image_path: Path, method: str = 'watershed') -> dict:
        """
        Process single image with watershed segmentation.
        
        Args:
            image_path: Path to image
            method: Always uses 'watershed' for this class
            
        Returns:
            Results dictionary
        """
        try:
            # Extract genotype info
            genotype_id, replicate = self.extract_genotype_info(image_path)
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'error': 'Failed to load image', 'success': False}
            
            # Segment with watershed
            results = self.watershed_segmenter.segment_overlapping_seeds(
                image, 
                threshold_method='adaptive',
                visualize=False
            )
            
            # Extract features from properties
            props = results['properties']
            
            if len(props) == 0:
                mean_area = 0
                std_area = 0
                mean_circularity = 0
                mean_aspect_ratio = 0
                mean_solidity = 0
                total_area = 0
            else:
                areas = [p.area for p in props]
                circularities = [4 * np.pi * p.area / (p.perimeter**2 + 1e-6) for p in props]
                aspect_ratios = [p.major_axis_length / (p.minor_axis_length + 1e-6) for p in props]
                solidities = [p.solidity for p in props]
                
                mean_area = np.mean(areas)
                std_area = np.std(areas)
                mean_circularity = np.mean(circularities)
                mean_aspect_ratio = np.mean(aspect_ratios)
                mean_solidity = np.mean(solidities)
                total_area = np.sum(areas)
            
            # Compile results
            output = {
                'genotype': genotype_id,
                'replicate': replicate,
                'image_name': image_path.name,
                'image_path': str(image_path),
                'seed_count': results['seed_count'],
                'mean_area': mean_area,
                'std_area': std_area,
                'mean_circularity': mean_circularity,
                'mean_aspect_ratio': mean_aspect_ratio,
                'mean_solidity': mean_solidity,
                'total_seed_area': total_area,
                'coverage_percent': (total_area / (image.shape[0] * image.shape[1]) * 100),
                'success': True
            }
            
            return output
            
        except Exception as e:
            return {
                'genotype': 'unknown',
                'replicate': 0,
                'image_name': image_path.name,
                'error': str(e),
                'success': False
            }


def auto_tune_parameters(image_path: str) -> dict:
    """
    Automatically find best watershed parameters for sample image.
    
    Args:
        image_path: Path to test image
        
    Returns:
        Dictionary with best parameters
    """
    print("\nAuto-tuning watershed parameters...")
    print("Testing different parameter combinations...")
    
    segmenter = OverlappingSeedsSegmenter()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load {image_path}")
        return {'min_distance': 15, 'sensitivity': 0.5}
    
    best_params = segmenter.tune_parameters(
        image,
        min_distance_range=(10, 30),
        sensitivity_range=(0.3, 0.7)
    )
    
    print(f"\n✓ Best parameters found:")
    print(f"  min_distance: {best_params['min_distance']}")
    print(f"  sensitivity: {best_params['sensitivity']:.2f}")
    print(f"  Detected seeds: {best_params['seed_count']}")
    
    return best_params


def quick_test(image_path: str, min_distance: int = 15, sensitivity: float = 0.5):
    """
    Quick test on single image.
    
    Args:
        image_path: Path to test image
        min_distance: Minimum distance parameter
        sensitivity: Sensitivity parameter
    """
    print(f"\nTesting watershed segmentation on: {image_path}")
    print(f"Parameters: min_distance={min_distance}, sensitivity={sensitivity}")
    print("="*60)
    
    segmenter = OverlappingSeedsSegmenter(
        min_distance=min_distance,
        sensitivity=sensitivity
    )
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image")
        return
    
    results = segmenter.segment_overlapping_seeds(image, visualize=True)
    
    print(f"\n✓ Detected {results['seed_count']} seeds")
    print(f"  Binary mask shape: {results['binary_mask'].shape}")
    print(f"  Labeled regions: {results['labeled_mask'].max()}")
    
    # Save visualization
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_segmented.png"
    cv2.imwrite(str(output_path), results['visualization'])
    print(f"  Visualization saved to: {output_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process rice genotypes with watershed segmentation for overlapping seeds'
    )
    parser.add_argument('--input', type=str, default='../data/raw',
                       help='Input directory with images')
    parser.add_argument('--output', type=str, default='../results/genotype_screening_watershed',
                       help='Output directory')
    parser.add_argument('--min-distance', type=int, default=15,
                       help='Minimum distance between seeds (10-30, lower=more sensitive)')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                       help='Watershed sensitivity (0.3-0.7, lower=more aggressive)')
    parser.add_argument('--pattern', type=str, default='*.jpg',
                       help='File pattern to match')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top genotypes to highlight')
    parser.add_argument('--test', type=str,
                       help='Test on single image first')
    parser.add_argument('--auto-tune', type=str,
                       help='Auto-tune parameters on sample image')
    parser.add_argument('--visualize-first', type=int,
                       help='Visualize first N images for quality check')
    
    args = parser.parse_args()
    
    # Test mode
    if args.test:
        quick_test(args.test, args.min_distance, args.sensitivity)
        return
    
    # Auto-tune mode
    if args.auto_tune:
        best_params = auto_tune_parameters(args.auto_tune)
        print(f"\nRecommended command:")
        print(f"python {Path(__file__).name} --input {args.input} "
              f"--min-distance {best_params['min_distance']} "
              f"--sensitivity {best_params['sensitivity']:.2f}")
        return
    
    # Full processing
    print("\n" + "="*70)
    print("GENOTYPE SCREENING WITH WATERSHED SEGMENTATION")
    print("="*70)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Parameters:")
    print(f"  min_distance: {args.min_distance} (lower = more sensitive)")
    print(f"  sensitivity: {args.sensitivity:.2f} (lower = more aggressive)")
    print("="*70)
    
    start_time = datetime.now()
    
    # Initialize processor with watershed
    processor = WatershedGenotypeProcessor(
        output_dir=args.output,
        min_distance=args.min_distance,
        sensitivity=args.sensitivity
    )
    
    # Find all images
    input_path = Path(args.input)
    image_files = list(input_path.rglob(args.pattern))
    image_files.extend(input_path.rglob(args.pattern.replace('.jpg', '.JPG')))
    image_files.extend(input_path.rglob(args.pattern.replace('.jpg', '.png')))
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"\nError: No images found in {args.input}")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    # Optional: Visualize first N for quality check
    if args.visualize_first:
        print(f"\nGenerating visualizations for first {args.visualize_first} images...")
        vis_dir = Path(args.output) / 'quality_check'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in image_files[:args.visualize_first]:
            image = cv2.imread(str(img_path))
            results = processor.watershed_segmenter.segment_overlapping_seeds(
                image, visualize=True
            )
            
            vis_path = vis_dir / f"{img_path.stem}_check.png"
            cv2.imwrite(str(vis_path), results['visualization'])
            print(f"  {img_path.name}: {results['seed_count']} seeds")
        
        print(f"\n✓ Visualizations saved to {vis_dir}")
        print("\nPlease check the quality. If segmentation looks good, run again without --visualize-first")
        return
    
    # Process all images
    print("\nProcessing all images with watershed segmentation...")
    df_individual = processor.process_all_images(
        Path(args.input),
        method='watershed',
        file_pattern=args.pattern
    )
    
    if df_individual.empty:
        print("No data to analyze!")
        return
    
    # Calculate statistics
    print("\nCalculating genotype statistics...")
    df_summary = processor.calculate_genotype_statistics(df_individual)
    
    # Create visualizations
    print("\nGenerating plots...")
    processor.create_ranking_plot(df_summary, top_n=args.top_n)
    processor.create_distribution_plots(df_individual, df_summary)
    
    # Print report
    processor.print_summary_report(df_summary, top_n=args.top_n)
    
    # Processing time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total processing time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Average time per image: {duration/len(df_individual):.2f} seconds")
    print(f"Images per second: {len(df_individual)/duration:.2f}")
    print(f"\nResults saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
