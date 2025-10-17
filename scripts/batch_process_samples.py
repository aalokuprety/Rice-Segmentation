#!/usr/bin/env python3
"""
Batch Process Rice Samples

Process all rice images in variety folders with automated segmentation.
Handles 100+ samples efficiently with progress tracking and error handling.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from rice_segmentation_methods import RiceSeedSegmenter
from evaluation_metrics import SegmentationEvaluator


class BatchProcessor:
    """Process multiple rice images with automated segmentation."""
    
    def __init__(self, output_dir: str = '../results'):
        """
        Initialize batch processor.
        
        Args:
            output_dir: Directory to save results
        """
        self.segmenter = RiceSeedSegmenter()
        self.evaluator = SegmentationEvaluator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'processed_images').mkdir(exist_ok=True)
        (self.output_dir / 'csv_reports').mkdir(exist_ok=True)
    
    def process_single_image(self, image_path: Path, method: str = 'adaptive') -> Dict:
        """
        Process a single image.
        
        Args:
            image_path: Path to image file
            method: Segmentation method to use
            
        Returns:
            Dictionary with results
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'error': 'Failed to load image'}
            
            # Select segmentation method
            if method == 'otsu':
                results = self.segmenter.otsu_segmentation(image)
            elif method == 'adaptive':
                results = self.segmenter.adaptive_segmentation(image)
            elif method == 'watershed':
                results = self.segmenter.watershed_segmentation(image)
            elif method == 'connected':
                results = self.segmenter.connected_components_segmentation(image)
            else:
                results = self.segmenter.adaptive_segmentation(image)  # default
            
            # Extract features
            features = self.segmenter.extract_features(results['labeled_mask'])
            
            # Compile results
            output = {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'method': method,
                'seed_count': results['seed_count'],
                'mean_area': features['area'].mean() if len(features) > 0 else 0,
                'std_area': features['area'].std() if len(features) > 0 else 0,
                'mean_perimeter': features['perimeter'].mean() if len(features) > 0 else 0,
                'mean_aspect_ratio': features['aspect_ratio'].mean() if len(features) > 0 else 0,
                'mean_circularity': features['circularity'].mean() if len(features) > 0 else 0,
                'mean_solidity': features['solidity'].mean() if len(features) > 0 else 0,
                'total_seed_area': features['area'].sum() if len(features) > 0 else 0,
                'coverage_percentage': (features['area'].sum() / (image.shape[0] * image.shape[1]) * 100) if len(features) > 0 else 0,
                'processing_time': results.get('processing_time', 0),
                'success': True
            }
            
            # Store masks for visualization
            output['binary_mask'] = results['binary_mask']
            output['labeled_mask'] = results['labeled_mask']
            output['features_df'] = features
            
            return output
            
        except Exception as e:
            return {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'error': str(e),
                'success': False
            }
    
    def process_variety(self, variety_dir: Path, method: str = 'adaptive', 
                       visualize: bool = False, max_images: Optional[int] = None) -> pd.DataFrame:
        """
        Process all images in a variety folder.
        
        Args:
            variety_dir: Path to variety folder
            method: Segmentation method
            visualize: Whether to save visualization images
            max_images: Maximum number of images to process (None = all)
            
        Returns:
            DataFrame with results
        """
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(variety_dir.glob(f'*{ext}'))
            image_files.extend(variety_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {variety_dir}")
            return pd.DataFrame()
        
        if max_images:
            image_files = image_files[:max_images]
        
        variety_name = variety_dir.name
        print(f"\nProcessing {variety_name}: {len(image_files)} images")
        print(f"Method: {method}")
        print("="*60)
        
        results = []
        for img_path in tqdm(image_files, desc=f"Processing {variety_name}"):
            result = self.process_single_image(img_path, method)
            
            if result.get('success', False):
                # Save visualization if requested
                if visualize:
                    self.save_visualization(img_path, result, variety_name)
                
                # Remove large arrays before storing in DataFrame
                result_clean = {k: v for k, v in result.items() 
                              if k not in ['binary_mask', 'labeled_mask', 'features_df']}
                results.append(result_clean)
            else:
                print(f"\nError processing {img_path.name}: {result.get('error', 'Unknown error')}")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        df['variety'] = variety_name
        
        # Save results
        csv_path = self.output_dir / 'csv_reports' / f'{variety_name}_{method}_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
        
        # Print summary
        self.print_variety_summary(df, variety_name)
        
        return df
    
    def save_visualization(self, image_path: Path, results: Dict, variety_name: str):
        """Save visualization of segmentation results."""
        try:
            # Load original image
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            overlay = image_rgb.copy()
            labeled_mask = results['labeled_mask']
            
            # Color each seed differently
            from skimage import color
            colored_labels = color.label2rgb(labeled_mask, image_rgb, alpha=0.3, bg_label=0)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original
            axes[0].imshow(image_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Segmentation overlay
            axes[1].imshow(colored_labels)
            axes[1].set_title(f'Segmentation ({results["seed_count"]} seeds)')
            axes[1].axis('off')
            
            # Binary mask
            axes[2].imshow(results['binary_mask'], cmap='gray')
            axes[2].set_title('Binary Mask')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save
            vis_dir = self.output_dir / 'visualizations' / variety_name
            vis_dir.mkdir(parents=True, exist_ok=True)
            save_path = vis_dir / f'{image_path.stem}_segmented.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create visualization for {image_path.name}: {e}")
    
    def print_variety_summary(self, df: pd.DataFrame, variety_name: str):
        """Print summary statistics for a variety."""
        print(f"\n{'='*60}")
        print(f"Summary: {variety_name}")
        print(f"{'='*60}")
        print(f"Images processed: {len(df)}")
        print(f"Total seeds detected: {df['seed_count'].sum():.0f}")
        print(f"Mean seeds per image: {df['seed_count'].mean():.1f} ± {df['seed_count'].std():.1f}")
        print(f"Range: {df['seed_count'].min():.0f} - {df['seed_count'].max():.0f} seeds")
        print(f"Mean seed area: {df['mean_area'].mean():.1f} pixels²")
        print(f"Mean circularity: {df['mean_circularity'].mean():.3f}")
        print(f"Mean coverage: {df['coverage_percentage'].mean():.1f}%")
        print(f"{'='*60}\n")
    
    def process_all_varieties(self, base_dir: Path, varieties: List[str], 
                             method: str = 'adaptive', visualize: bool = False,
                             max_images_per_variety: Optional[int] = None) -> pd.DataFrame:
        """
        Process all varieties.
        
        Args:
            base_dir: Base directory containing variety folders
            varieties: List of variety folder names
            method: Segmentation method
            visualize: Whether to save visualizations
            max_images_per_variety: Max images per variety
            
        Returns:
            Combined DataFrame with all results
        """
        all_results = []
        
        for variety in varieties:
            variety_dir = base_dir / variety
            if variety_dir.exists():
                df = self.process_variety(variety_dir, method, visualize, max_images_per_variety)
                if not df.empty:
                    all_results.append(df)
            else:
                print(f"Warning: Directory not found: {variety_dir}")
        
        if not all_results:
            print("No results to combine!")
            return pd.DataFrame()
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_path = self.output_dir / 'csv_reports' / f'all_varieties_{method}_{timestamp}.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"\n✓ Combined results saved to {combined_path}")
        
        # Create comparison report
        self.create_comparison_report(combined_df, method)
        
        return combined_df
    
    def create_comparison_report(self, df: pd.DataFrame, method: str):
        """Create visual comparison report for all varieties."""
        varieties = df['variety'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Variety Comparison - {method.title()} Segmentation', fontsize=16, fontweight='bold')
        
        # 1. Seed count distribution
        ax = axes[0, 0]
        df.boxplot(column='seed_count', by='variety', ax=ax)
        ax.set_title('Seed Count Distribution')
        ax.set_xlabel('Variety')
        ax.set_ylabel('Seed Count')
        plt.sca(ax)
        plt.xticks(rotation=45)
        
        # 2. Mean seed area
        ax = axes[0, 1]
        variety_means = df.groupby('variety')['mean_area'].agg(['mean', 'std'])
        ax.bar(variety_means.index, variety_means['mean'], 
               yerr=variety_means['std'], capsize=5, alpha=0.7)
        ax.set_title('Mean Seed Area')
        ax.set_xlabel('Variety')
        ax.set_ylabel('Area (pixels²)')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Circularity comparison
        ax = axes[1, 0]
        for variety in varieties:
            variety_data = df[df['variety'] == variety]['mean_circularity']
            ax.hist(variety_data, alpha=0.5, label=variety, bins=20)
        ax.set_title('Seed Circularity Distribution')
        ax.set_xlabel('Circularity')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # 4. Summary table
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_data = []
        for variety in varieties:
            variety_df = df[df['variety'] == variety]
            summary_data.append([
                variety,
                f"{len(variety_df)}",
                f"{variety_df['seed_count'].mean():.1f}",
                f"{variety_df['seed_count'].std():.1f}",
                f"{variety_df['mean_area'].mean():.0f}"
            ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Variety', 'Images', 'Mean Count', 'Std', 'Mean Area'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = self.output_dir / f'variety_comparison_{method}_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {save_path}")
        plt.close()


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(description='Batch process rice samples')
    parser.add_argument('--base-dir', type=str, default='../data/raw',
                       help='Base directory containing variety folders')
    parser.add_argument('--variety', type=str,
                       help='Process single variety (folder name)')
    parser.add_argument('--all-varieties', action='store_true',
                       help='Process all varieties')
    parser.add_argument('--method', type=str, default='adaptive',
                       choices=['otsu', 'adaptive', 'watershed', 'connected'],
                       help='Segmentation method to use')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--max-images', type=int,
                       help='Maximum images to process per variety (for testing)')
    parser.add_argument('--output-dir', type=str, default='../results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchProcessor(output_dir=args.output_dir)
    base_path = Path(args.base_dir)
    
    print("\n" + "="*70)
    print("RICE SAMPLE BATCH PROCESSOR")
    print("="*70)
    print(f"Base directory: {base_path}")
    print(f"Segmentation method: {args.method}")
    print(f"Visualizations: {'Enabled' if args.visualize else 'Disabled'}")
    if args.max_images:
        print(f"Max images per variety: {args.max_images}")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    if args.variety:
        # Process single variety
        variety_dir = base_path / args.variety
        if not variety_dir.exists():
            print(f"Error: Directory not found: {variety_dir}")
            return
        
        df = processor.process_variety(variety_dir, args.method, 
                                      args.visualize, args.max_images)
    
    elif args.all_varieties:
        # Find all variety folders
        variety_folders = [d.name for d in base_path.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')]
        
        if not variety_folders:
            print(f"No variety folders found in {base_path}")
            return
        
        print(f"Found varieties: {', '.join(variety_folders)}\n")
        
        df = processor.process_all_varieties(base_path, variety_folders, 
                                            args.method, args.visualize,
                                            args.max_images)
    else:
        print("Error: Please specify --variety or --all-varieties")
        parser.print_help()
        return
    
    # Print final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total images processed: {len(df)}")
    print(f"Total seeds detected: {df['seed_count'].sum():.0f}")
    print(f"Processing time: {duration:.1f} seconds")
    print(f"Average time per image: {duration/len(df):.2f} seconds")
    print(f"Results saved to: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()